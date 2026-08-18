import os
import sys
import traceback
from collections import defaultdict
import filecmp

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_auroc, binary_recall, binary_f1_score, binary_precision
from torchmetrics import AUROC, Recall, F1Score, Precision, Accuracy

import matplotlib.pyplot as plt
from tqdm import tqdm
import mir_eval
import pretty_midi
import glob

from utils import seed_everything
from utils.commons.hparams import hparams
from utils.commons.base_task import BaseTask
from utils.audio.pitch_utils import denorm_f0, boundary2Interval, midi_to_hz, save_midi, \
    validate_pitch_and_itv, midi_melody_eval, melody_eval_pitch_and_itv, validate_itv
from utils.commons.dataset_utils import data_loader, BaseConcatDataset, build_dataloader
from utils.commons.tensor_utils import tensors_to_scalars
from utils.commons.ckpt_utils import load_ckpt
from utils.nn.model_utils import print_arch
from utils.commons.multiprocess_utils import MultiprocessManager
from utils.audio.pitch_utils import midi_onset_eval, midi_offset_eval, midi_pitch_eval, \
    midi_COn_eval, midi_COnP_eval, midi_COnPOff_eval, midi2NoteInterval, midi2NotePitch
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from utils.commons.losses import sigmoid_focal_loss
from utils.nn.schedulers import RSQRTSchedule, NoneSchedule, WarmupSchedule
from tasks.stars.dataset import StarsDataset, PhoneEncoder
from tasks.stars.utils import bd_to_durs, bd_to_idxs, regulate_ill_slur, regulate_real_note_itv
from modules.stars.stars import STARS

import textgrid
from textgrid import PointTier
from utils.metrics.align_metrics import (
    BoundaryEditRatio,
    IntersectionOverUnion,
    Metric,
    VlabelerEditRatio,
    StyleAcc
)
def run_viterbi_core(dp_matrix, backtrace_dp_matrix, cur_log_prediction, cur_log_silence_prediction, cur_label, bd_log_prediction, not_bd_log_prediction):
    # print (cur_label.shape[0] * 2 + 1, log_prediction.shape[1])
    for j in range(1, cur_log_prediction.shape[0]):
        for k in range(cur_label.shape[0] * 2 + 1):
            if k == 0:
                # blank
                backtrace_dp_matrix[j][k] = k
                dp_matrix[j][k] = dp_matrix[j-1][k] + cur_log_silence_prediction[j] + not_bd_log_prediction[j]

            elif k == 1:
                if dp_matrix[j-1][k] + not_bd_log_prediction[j] > dp_matrix[j-1][k-1] + bd_log_prediction[j]:
                    backtrace_dp_matrix[j][k] = k
                    dp_matrix[j][k] = dp_matrix[j-1][k] + cur_log_prediction[j][cur_label[0] - 2] + not_bd_log_prediction[j]
                else:
                    backtrace_dp_matrix[j][k] = k - 1
                    dp_matrix[j][k] = dp_matrix[j-1][k-1] + cur_log_prediction[j][cur_label[0] - 2] + bd_log_prediction[j]

            elif k % 2 == 0:
                # blank
                if dp_matrix[j-1][k] + not_bd_log_prediction[j] > dp_matrix[j-1][k-1] + bd_log_prediction[j]:
                    backtrace_dp_matrix[j][k] = k
                    dp_matrix[j][k] = dp_matrix[j-1][k] + cur_log_silence_prediction[j] + not_bd_log_prediction[j]
                else:
                    backtrace_dp_matrix[j][k] = k - 1
                    dp_matrix[j][k] = dp_matrix[j-1][k-1] + cur_log_silence_prediction[j] + bd_log_prediction[j]

            else:
                if dp_matrix[j-1][k-2] + bd_log_prediction[j] >= dp_matrix[j-1][k-1] + bd_log_prediction[j] and dp_matrix[j-1][k-2] + bd_log_prediction[j] >= dp_matrix[j-1][k] + not_bd_log_prediction[j]:
                    # k-2 (last character) -> k
                    backtrace_dp_matrix[j][k] = k - 2
                    dp_matrix[j][k] = dp_matrix[j-1][k-2] + cur_log_prediction[j][cur_label[k // 2] - 2] + bd_log_prediction[j]

                elif dp_matrix[j-1][k] + not_bd_log_prediction[j] > dp_matrix[j-1][k-1] + bd_log_prediction[j]:
                    # k -> k
                    backtrace_dp_matrix[j][k] = k
                    dp_matrix[j][k] = dp_matrix[j-1][k] + cur_log_prediction[j][cur_label[k // 2] - 2] + not_bd_log_prediction[j]
                else:
                    # k-1 -> k
                    backtrace_dp_matrix[j][k] = k - 1
                    dp_matrix[j][k] = dp_matrix[j-1][k-1] + cur_log_prediction[j][cur_label[k // 2] - 2] + bd_log_prediction[j]

    return dp_matrix, backtrace_dp_matrix

def get_textgrid(mel_len, ph_of, word_of, tech_pred=None, hop_size_second=0.02, draw=False, id2token=None, word_list=None, dp_matrix=None, mel=None, tg_fn=None):
    minTime = 0
    maxTime = max(mel_len.cpu().numpy(), word_of[-1][1], ph_of[-1][1]) * hop_size_second
    tg = textgrid.TextGrid(minTime=minTime, maxTime=maxTime)
    word_intervals = textgrid.IntervalTier(name="words", minTime=minTime, maxTime=maxTime)
    phone_intervals = textgrid.IntervalTier(name="phones", minTime=minTime, maxTime=maxTime)
    if tech_pred!=None:
        tech_intervals = {}
        tech_names = ['bubble', 'breathe', 'pharyngeal', 'vibrato', 'glissando', 'mixed', 'falsetto', 'weak', 'strong']
        for i, name in enumerate(tech_names):
            tech_intervals[name] = textgrid.IntervalTier(name=name, minTime=minTime, maxTime=maxTime)

    for word in word_of:
        word_id = word[2]
        minTime = word[0] * hop_size_second
        maxTime = word[1] * hop_size_second
        if word_id == -1:
            mark = '<SP>'
        else:
            mark = word_list[word_id]
        tg_word = textgrid.Interval(minTime=minTime, maxTime=maxTime, mark=mark)
        word_intervals.addInterval(tg_word)

    for idx, phone in enumerate(ph_of):
        ph_id = phone[2]
        minTime = phone[0] * hop_size_second
        maxTime = phone[1] * hop_size_second
        mark = id2token[ph_id]
        tg_phone = textgrid.Interval(minTime=minTime, maxTime=maxTime, mark=mark)
        phone_intervals.addInterval(tg_phone)
        if tech_pred!=None:
            for i, name in enumerate(tech_names):
                tg_tech = textgrid.Interval(minTime=minTime, maxTime=maxTime, mark=str(tech_pred[idx][i].item()))
                tech_intervals[name].addInterval(tg_tech)

    tg.tiers.append(word_intervals)
    tg.tiers.append(phone_intervals)
    if tech_pred!=None:
        for i, name in enumerate(tech_names):
            tg.tiers.append(tech_intervals[name])

    tg.write(tg_fn)

    if draw: 
        plt.style.use('default')
        fig = plt.figure(figsize=(18, 12), facecolor='white')
        gs = fig.add_gridspec(3, 1, height_ratios=[0.2, 2, 1], hspace=0.08)

        ax_title = fig.add_subplot(gs[0])
        ax_mel = fig.add_subplot(gs[1])
        ax_dp = fig.add_subplot(gs[2])

        ax_title.axis('off')
        
        if mel is not None:
            mel = mel.numpy() if isinstance(mel, torch.Tensor) else mel
            time_steps, n_mels = mel.shape
            
            im_mel = ax_mel.imshow(mel.T, origin='lower', aspect='auto',
                                cmap='viridis', interpolation='nearest')
            
            cbar = plt.colorbar(im_mel, ax=ax_mel, fraction=0.02, pad=0.01)
            cbar.set_label('Energy (dB)', rotation=270, labelpad=15)

            title_text = f"Phoneme Alignment Visualization"
            ax_title.text(0.5, 0.5, title_text, 
                        ha='center', va='center', 
                        fontsize=14, color='navy')

            label_params = {
                'base_y': n_mels * 1.00,
                'line_height': - n_mels * 0.04,
                'max_lines': 2,  
                'char_width': 0.1,  
                'min_gap': 0.15,
                'fontsize': 10,
            }

            def format_label(ph_id):
                text = id2token[ph_id]
                return text.replace('<', '').replace('>', '')

            sorted_ph = sorted(ph_of, key=lambda x: x[0])
            timeline = [[] for _ in range(label_params['max_lines'])]
            
            for idx, ph in enumerate(sorted_ph):
                start, end, ph_id, _ = ph
                text = format_label(ph_id)
                mid = (start + end) / 2
                
                if all(abs(mid - prev_mid) > label_params['min_gap'] for prev_mid in timeline[0]):
                    current_line = 0
                    timeline[current_line].append(mid)
                else:
                    current_line = 1
                    timeline[current_line].append(mid)

                y_pos = label_params['base_y'] + current_line * label_params['line_height']
                
                ax_mel.text(mid, y_pos, text,
                            color='darkred',
                            ha='center', va='bottom',
                            fontsize=label_params['fontsize'], weight='bold',
                            bbox=dict(facecolor='white', alpha=0.9,
                                    edgecolor='none', lw=0.3,
                                    boxstyle='round,pad=0.1'))

                ax_mel.axvline(start, color='white', ls='-', lw=0.8, alpha=0.8)
                ax_mel.axvline(end, color='white', ls='-', lw=0.8, alpha=0.8)

        dp_matrix_np = np.array(dp_matrix)
        
        valid_mask = dp_matrix_np > -1e7
        dp_valid = np.where(valid_mask, dp_matrix_np, np.nan)
        
        vmin = np.nanpercentile(dp_valid, 5)
        vmax = np.nanpercentile(dp_valid, 95)
        
        im_dp = ax_dp.imshow(dp_valid.T, aspect='auto', origin='lower',
                            cmap='plasma', vmin=vmin, vmax=vmax,
                            interpolation='nearest')
        
        cbar_dp = plt.colorbar(im_dp, ax=ax_dp, fraction=0.02, pad=0.01)
        cbar_dp.set_label('Log Probability', rotation=270, labelpad=15)
        
        prev_state_pos = 0
        num_phone = 0
        for ph_idx, ph in enumerate(ph_of):
            start_frame, end_frame, ph_id, _ = ph
            if ph_id != 1:
                state_pos = num_phone * 2 + 1
                num_phone += 1
            else:
                state_pos = num_phone * 2
            
            ax_dp.hlines(state_pos, start_frame, end_frame,
                        colors='darkred', linewidths=2, 
                        alpha=0.9, linestyle='-')
            
            ax_dp.vlines(start_frame, prev_state_pos-0.25, state_pos+0.25, 
                        colors='darkred', linewidths=2, 
                        alpha=0.9, linestyle='-')
            
            prev_state_pos = state_pos
        time_formatter = lambda x, _: f"{x*hop_size_second:.2f}s"
        
        ax_mel.set_xlim(0, dp_matrix_np.shape[0])
        ax_mel.xaxis.set_major_formatter(time_formatter)
        ax_mel.set_ylabel("Mel Bins")

        ax_dp.set_xlabel("Time (seconds)")
        ax_dp.set_ylabel("Viterbi States")
        ax_dp.xaxis.set_major_formatter(time_formatter)

        output_img_path = tg_fn.replace('.TextGrid', '_alignment.png')
        plt.savefig(output_img_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Visualization saved to {output_img_path}")
        plt.close()
    return True

def onset_offset_to_point_tier(ph_of, hop_size_second=0.02, unknow_mark=0) -> PointTier:
    point_tier = PointTier(name='phone')
    point_tier.add(0.0, unknow_mark)
    for phone in ph_of:
        if point_tier[-1].mark==unknow_mark and point_tier[-1].time==phone[0] * hop_size_second:
            point_tier[-1].mark = phone[2]
        else:
            point_tier.add(phone[0] * hop_size_second, phone[2])
        point_tier.add(phone[1] * hop_size_second, unknow_mark)
    return point_tier

def remove_ignored_phonemes(ignored_phonemes_list, point_tier: PointTier):
    res_tier = PointTier(name=point_tier.name)
    for i in range(len(point_tier) - 1):
        if point_tier[i].mark in ignored_phonemes_list:
            continue
        res_tier.addPoint(point_tier[i])
    res_tier.addPoint(point_tier[-1])
    return res_tier

def parse_dataset_configs():
    max_tokens = hparams['max_tokens']
    max_sentences = hparams['max_sentences']
    max_valid_tokens = hparams['max_valid_tokens']
    if max_valid_tokens == -1:
        hparams['max_valid_tokens'] = max_valid_tokens = max_tokens
    max_valid_sentences = hparams['max_valid_sentences']
    if max_valid_sentences == -1:
        hparams['max_valid_sentences'] = max_valid_sentences = max_sentences
    return max_tokens, max_sentences, max_valid_tokens, max_valid_sentences

class StarsTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_cls = StarsDataset
        self.vocoder = None
        self.saving_result_pool = None
        self.saving_results_futures = None
        self.max_tokens, self.max_sentences, \
            self.max_valid_tokens, self.max_valid_sentences = parse_dataset_configs()
        seed_everything(hparams['seed'])
        self.metrics = None
        self.stylemetrics = None
        self.ph_encoder = PhoneEncoder(os.path.join(hparams["processed_data_dir"], "phone_set.json"))

    @data_loader
    def train_dataloader(self):
        train_dataset = self.dataset_cls(prefix=hparams['train_set_name'], shuffle=True)
        return build_dataloader(train_dataset, True, self.max_tokens, self.max_sentences, endless=hparams['endless_ds'],
                                pin_memory=hparams.get('pin_memory', False), use_ddp=self.trainer.use_ddp)

    @data_loader
    def val_dataloader(self):
        valid_dataset = self.dataset_cls(prefix=hparams['valid_set_name'], shuffle=False)
        return build_dataloader(valid_dataset, False, self.max_valid_tokens, self.max_valid_sentences,
                                apply_batch_by_size=False, pin_memory=hparams.get('pin_memory', False),
                                use_ddp=self.trainer.use_ddp)

    @data_loader
    def test_dataloader(self):
        test_dataset = self.dataset_cls(prefix=hparams['test_set_name'], shuffle=False)
        self.test_dl = build_dataloader(
            test_dataset, False, self.max_valid_tokens, self.max_valid_sentences,
            apply_batch_by_size=False, pin_memory=hparams.get('pin_memory', False), use_ddp=self.trainer.use_ddp)
        return self.test_dl

    def build_model(self):
        self.model = STARS(hparams)
        # if hparams['load_ckpt'] != '':
        #     load_ckpt(self.model, hparams['load_ckpt'])
        print_arch(self.model)
        return self.model

    def build_scheduler(self, optimizer):
        last_step = max(-1, self.global_step-hparams.get('accumulate_grad_batches', 1))
        if hparams['scheduler'] == 'rsqrt':
            return RSQRTSchedule(optimizer, hparams['lr'], round(hparams['warmup_updates'] / hparams.get('accumulate_grad_batches', 1)), hparams['hidden_size'], last_step=last_step)
        elif hparams['scheduler'] == 'warmup':
            return WarmupSchedule(optimizer, hparams['lr'], hparams['warmup_updates'] / hparams.get('accumulate_grad_batches', 1), last_step=last_step)
        elif hparams['scheduler'] == 'step_lr':
            return torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=round(hparams.get('scheduler_lr_step_size', 500) / hparams.get('accumulate_grad_batches', 1)),
                gamma=hparams.get('scheduler_lr_gamma', 0.998), last_epoch=last_step)
        else:
            return NoneSchedule(optimizer, hparams['lr'])

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(
            [{'params': model.parameters(), 'initial_lr': hparams['lr']}],
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])

        return optimizer

    def _training_step(self, sample, batch_idx, _):
        loss_output, _ = self.run_model(sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['nsamples']
        return total_loss, loss_output

    def run_model(self, sample, infer=False):        
        mel = sample['mels']
        word_bd = sample['word_bd']
        ph_bd = sample['ph_bd']
        notes = sample['notes']
        note_bd = sample['note_bd']     # [B, T]
        ph_bd_soft = sample['ph_bd_soft']
        ph_bd = sample['ph_bd']
        pitch_coarse = sample['pitch_coarse']
        ph = sample['ph']
        uv = sample['uv'].long()
        mel_nonpadding = sample['mel_nonpadding']

        technique_target = sample['tech_ids']
        language_target = sample['languages']
        gender_target = sample['genders']
        emotion_target = sample['emotions']
        method_target = sample['singing_methods']
        pace_target = sample['paces']
        range_target = sample['ranges']
        techs = sample['techs']
        ph_lengths = sample['ph_lengths']
        mel_lengths = sample['mel_lengths']

        ph_nonpadding = sample['ph_nonpadding']
        word_nonpadding = sample['word_nonpadding']
        note_nonpadding = sample['note_nonpadding']
        mel2ph = sample['mel2ph']
        mel2word = sample['mel2word']
        mel2note = sample['mel2note']
        ph2words = sample['ph2words']

        output = self.model(mel=mel, pitch=pitch_coarse, uv=uv, mel_nonpadding=mel_nonpadding, ph_nonpadding=ph_nonpadding, word_nonpadding=word_nonpadding, note_nonpadding=note_nonpadding,
                            mel2ph=mel2ph, mel2word=mel2word, mel2note=mel2note, note_bd=note_bd, ph_bd=ph_bd, ph=ph, ph_lengths=ph_lengths, mel_lengths=mel_lengths, ph2words=ph2words, train=not infer, global_steps=self.global_step)

        losses = {}
        if not infer:
            # frame level phone loss
            if hparams.get('use_soft_ph_bd', False) and (hparams.get('soft_ph_bd_func', None) not in ['none', None]):
                self.add_ph_bd_loss(output['ph_bd_logits'], ph_bd_soft, losses)
            else:
                self.add_ph_bd_loss(output['ph_bd_logits'], ph_bd, losses)

            input_lengths = sample['mel_lengths']
            ph_frame = sample['ph_frame']
            ctc_logits = torch.cat([output['ph_frame_logits'][:, :, [0]], output['ph_frame_logits'][:, :, 2:]], dim=-1)

            if hparams.get('use_ctc_loss', True):
                self.add_phone_ctc_loss(ctc_logits, ph, input_lengths, ph_lengths, losses)
            self.add_phone_ce_loss(output['ph_frame_logits'][:,:,1:], ph_frame, losses)
            
            # frame level note loss
            if self.global_step >= hparams.get('note_bd_start', 0):
                note_bd_soft = sample['note_bd_soft']
                note_frame = sample['note_frame']
                self.add_note_bd_loss(output['note_bd_logits'], note_bd_soft.float(), losses)
                self.add_note_ce_loss(output['note_frame_logits'][:,:,1:], note_frame, losses)
            else:
                losses['note_bd'] = 0.
                losses['note_bd_fc'] = 0.
                losses['note_ce_loss'] = 0.
            
            if self.global_step >= hparams.get('note_pitch_start', 0):
                self.add_note_loss(output['note_logits'], notes, losses)
            else:
                losses['pitch'] = 0.0

            # phone level technique loss
            if self.global_step >= hparams.get('tech_start', 0):
                self.add_tech_loss(output['tech_logits'], techs, technique_target, ph_lengths, losses)
            else:
                tech_names = ['bubble', 'breathe', 'pharyngeal', 'vibrato', 'glissando', 'mixed', 'falsetto', 'weak', 'strong']
                for i, name in enumerate(tech_names):
                    losses[name] = 0.

            if self.global_step > hparams['vq_ph_start']:
                losses['vq_loss'] = output['vq_loss_ph'] / 3
                losses['ppl_ph'] = output['ppl_ph'].detach()

            if self.global_step > hparams['vq_word_start']:
                losses['vq_loss'] = losses['vq_loss'] + output['vq_loss_word'] / 3
                losses['ppl_word'] = output['ppl_word'].detach()
            
            if self.global_step > hparams['vq_note_start']:
                losses['vq_loss'] = losses['vq_loss'] + output['vq_loss_note'] / 3
                losses['ppl_note'] = output['ppl_note'].detach()

            # global style
            if self.global_step >= hparams.get('att_start', 0):
                losses['style'] = 0.
                losses['style'] += self.add_att_loss(output['technique_logits'], technique_target) / 7
                losses['style'] += self.add_att_loss(output['language_logits'], language_target) / 7
                losses['style'] += self.add_att_loss(output['gender_logits'], gender_target) / 7
                losses['style'] += self.add_att_loss(output['emotion_logits'], emotion_target) / 7
                losses['style'] += self.add_att_loss(output['method_logits'], method_target) / 7
                losses['style'] += self.add_att_loss(output['pace_logits'], pace_target) / 7
                losses['style'] += self.add_att_loss(output['range_logits'], range_target) / 7
            else:
                losses['style'] = 0.

        return losses, output
    
    def add_note_loss(self, note_logits, notes, losses):
        note_pitch_loss = F.cross_entropy(note_logits.transpose(1, 2), notes, label_smoothing=hparams.get('note_pitch_label_smoothing', 0.0))
        losses['pitch'] = note_pitch_loss * hparams.get('lambda_note_pitch', 1.0)

    def add_tech_loss(self, tech_logits, techs, tech_group, target_lengths, losses):
        bsz, T, binary_num_techs = tech_logits.shape
        lambda_tech = [1] * 10

        valid_mask = (techs != -1).float()  # [bsz, T, binary_num_techs]
        techs = techs * valid_mask

        all_losses = F.binary_cross_entropy_with_logits(tech_logits, techs.float(), reduction='none') * valid_mask  # [bsz, T, binary_num_techs]

        len_mask = (torch.arange(T, device=tech_logits.device).unsqueeze(0) < target_lengths.unsqueeze(1)).unsqueeze(-1)  # [bsz, T, 1]

        tech_nonzero = torch.any(techs != 0, dim=1)  # [bsz, binary_num_techs]

        tech_group_tensor = torch.tensor(
            [g if g is not None else -1 for g in tech_group],
            device=tech_logits.device
        )
        valid_mask = tech_group_tensor != -1 
        group0_mask = (tech_group_tensor == 0) & valid_mask
        non_group0_mask = (tech_group_tensor != 0) & valid_mask

        tech_losses = torch.zeros((T, binary_num_techs), device=tech_logits.device)
        tech_num = torch.zeros(binary_num_techs, device=tech_logits.device)

        if group0_mask.any():
            group0_loss = all_losses[group0_mask]
            group0_len_mask = len_mask[group0_mask].squeeze(-1)  # [num_group0, T]
            group0_tech_mask = tech_nonzero[group0_mask]  # [num_group0, binary_num_techs]
            
            masked_group0_loss = group0_loss * group0_tech_mask.unsqueeze(1) * group0_len_mask.unsqueeze(-1)
            tech_losses += masked_group0_loss.sum(dim=0) 
            
            group0_lens = target_lengths[group0_mask]
            tech_num += (group0_tech_mask.float().T @ group0_lens.float()).T

        if non_group0_mask.any():
            non_group0_loss = all_losses[non_group0_mask]
            non_group0_len_mask = len_mask[non_group0_mask].squeeze(-1)  # [num_non_group0, T]
            
            masked_non_group0_loss = non_group0_loss * non_group0_len_mask.unsqueeze(-1)
            tech_losses += masked_non_group0_loss.sum(dim=0)
            
            sum_non_group0_lens = target_lengths[non_group0_mask].sum()
            tech_num += sum_non_group0_lens

        tech_num = tech_num.clamp(min=1e-8)
        tech_losses = tech_losses.sum(dim=0) / tech_num

        tech_names = ['bubble', 'breathe', 'pharyngeal', 'vibrato', 'glissando', 'mixed', 'falsetto', 'weak', 'strong']
        for i, name in enumerate(tech_names):
            losses[name] = tech_losses[i] * lambda_tech[i]
            
    def add_att_loss(self, pred_logits, target):
        valid_indices = [i for i, val in enumerate(target) if val is not None]
        if len(valid_indices) == 0:
            return 0.
        if pred_logits.dim() == 3:
            pred_logits = pred_logits.squeeze(1)
        valid_pred_logits = pred_logits[valid_indices, :]
        valid_target = torch.tensor([target[i] for i in valid_indices], device=pred_logits.device)
        loss = F.cross_entropy(valid_pred_logits, valid_target, label_smoothing=hparams.get('att_label_smoothing', 0.0))
        return loss * hparams.get('lambda_att_ce', 1.0)

    def add_phone_ctc_loss(self, ph_frame_logits, ph, input_lengths, target_length, losses):
        if self.global_step >= hparams.get('ph_start', 0):
            ph_frame_logits = ph_frame_logits.permute(1, 0, 2)
            ph_log_probs = torch.nn.functional.log_softmax(ph_frame_logits, dim=-1)
            ph_ctc_loss = torch.nn.functional.ctc_loss(ph_log_probs, ph, input_lengths, target_length)
            losses['ph_ctc_loss'] = ph_ctc_loss * hparams.get('lambda_ph_ctc', 1.0)
        else:
            losses['ph_ctc_loss'] = 0.
            
    def add_ph_bd_loss(self, ph_bd_logits, ph_bd_soft, losses):
        if self.global_step >= hparams.get('ph_start', 0):
            if not hasattr(self, 'ph_bd_pos_weight'):
                self.ph_bd_pos_weight = torch.ones(5000).to(ph_bd_logits.device)    # cache
                if hparams.get('label_pos_weight_decay', 0.0) > 0.0:
                    ph_bd_ratio = hparams.get('ph_bd_ratio', 3) * hparams['hop_size'] / hparams['audio_sample_rate']
                    ph_bd_pos_weight = 1 / ph_bd_ratio
                    self.ph_bd_pos_weight = self.ph_bd_pos_weight * ph_bd_pos_weight * hparams.get('label_pos_weight_decay', 0.0)
            ph_bd_pos_weight = self.ph_bd_pos_weight[:ph_bd_logits.shape[1]]
            
            ph_bd_loss = F.binary_cross_entropy_with_logits(ph_bd_logits, ph_bd_soft, pos_weight=ph_bd_pos_weight)
            losses['ph_bd'] = ph_bd_loss * hparams.get('lambda_ph_bd', 1.0)
            # add focal loss
            if hparams.get('ph_bd_focal_loss', None) not in ['none', None, 0]:
                gamma = float(hparams.get('ph_bd_focal_loss', None))
                focal_loss = sigmoid_focal_loss(
                    ph_bd_logits, ph_bd_soft, alpha=1 / self.ph_bd_pos_weight[0], gamma=gamma, reduction='mean')
                losses['ph_bd_fc'] = focal_loss * hparams.get('lambda_ph_bd_focal', 1.0)
        else:
            losses['ph_bd'] = 0.0 * hparams.get('lambda_ph_bd', 1.0)
            if hparams.get('ph_bd_focal_loss', None) not in ['none', None]:
                losses['ph_bd_fc'] = 0.0 * hparams.get('lambda_ph_bd_focal', 1.0)

    def add_phone_ce_loss(self, ph_logits, ph, losses):
        if self.global_step >= hparams.get('ph_start', 0):
            ph_pitch_loss = F.cross_entropy(ph_logits.transpose(1, 2), ph, label_smoothing=hparams.get('ph_pitch_label_smoothing', 0.0))
            losses['ph_ce_loss'] = ph_pitch_loss * hparams.get('lambda_ph_ce', 1.0)
        else:
            losses['ph_ce_loss'] = 0.
            
    def add_note_bd_loss(self, note_bd_logits, note_bd, losses):
        if not hasattr(self, 'note_bd_pos_weight'):
            self.note_bd_pos_weight = torch.ones(5000).to(note_bd_logits.device)    # cache
            if hparams.get('label_pos_weight_decay', 0.0) > 0.0:
                note_bd_ratio = hparams.get('note_bd_ratio', 3) * hparams['hop_size'] / hparams['audio_sample_rate']
                note_bd_pos_weight = 1 / note_bd_ratio
                self.note_bd_pos_weight = self.note_bd_pos_weight * note_bd_pos_weight * hparams.get('label_pos_weight_decay', 0.0)
        note_bd_pos_weight = self.note_bd_pos_weight[:note_bd_logits.shape[1]]
        
        note_bd_loss = F.binary_cross_entropy_with_logits(note_bd_logits, note_bd, pos_weight=note_bd_pos_weight)
        losses['note_bd'] = note_bd_loss * hparams.get('lambda_note_bd', 1.0)
        # add focal loss
        if hparams.get('note_bd_focal_loss', None) not in ['none', None, 0]:
            gamma = float(hparams.get('note_bd_focal_loss', None))
            focal_loss = sigmoid_focal_loss(
                note_bd_logits, note_bd, alpha = 1 / self.note_bd_pos_weight[0], gamma=gamma, reduction='mean')
            losses['note_bd_fc'] = focal_loss * hparams.get('lambda_note_bd_focal', 1.0)

    def add_note_ce_loss(self, note_logits, note, losses):
        note_pitch_loss = F.cross_entropy(note_logits.transpose(1, 2), note, label_smoothing=hparams.get('note_pitch_label_smoothing', 0.0))
        losses['note_ce_loss'] = note_pitch_loss * hparams.get('lambda_note_ce', 1.0)
            
    def validation_start(self):
        pass

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        if 'mels' not in sample:
            return outputs
        with torch.no_grad():
            outputs['losses'], model_out = self.run_model(sample, infer=False)
        
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        if batch_idx < hparams['num_valid_stats']:
            mel = sample['mels']
            word_bd = sample['word_bd']
            ph_bd = sample['ph_bd']
            ph = sample['ph']
            ph_lengths = sample['ph_lengths']
            mel_lengths = sample['mel_lengths']
            pitch_coarse = sample['pitch_coarse']
            uv = sample['uv'].long()
            mel_nonpadding = sample['mel_nonpadding']
            ph_nonpadding = sample['ph_nonpadding']
            word_nonpadding = sample['word_nonpadding']
            mel2ph = sample['mel2ph']
            mel2word = sample['mel2word']
            gt_f0 = denorm_f0(sample['f0'], sample["uv"])
            bsz = sample['nsamples']
            note_nonpadding = sample['note_nonpadding']
            mel2note = sample['mel2note']
            note_bd = sample['note_bd']
            ph2words = sample['ph2words']
            tech_threshold = 0.8
            with torch.no_grad():
                metrics: Dict[str, Metric] = {
                    "BoundaryEditRatio": BoundaryEditRatio(),
                    "VlabelerEditRatio10ms": VlabelerEditRatio(move_tolerance=0.01),
                    "VlabelerEditRatio20ms": VlabelerEditRatio(move_tolerance=0.02),
                    "VlabelerEditRatio50ms": VlabelerEditRatio(move_tolerance=0.05),
                    "IntersectionOverUnion": IntersectionOverUnion(),
                }
                note_onset_f, note_offset_f, overlap_f, avg_overlap_ratio, rpa = 0, 0, 0, 0, 0
                sample_num = 0
                output = self.model(mel=mel, pitch=pitch_coarse, uv=uv, mel_nonpadding=mel_nonpadding, ph_nonpadding=ph_nonpadding, word_nonpadding=word_nonpadding, note_nonpadding=note_nonpadding,
                            mel2ph=mel2ph, mel2word=mel2word, mel2note=mel2note, note_bd=note_bd, ph_bd=ph_bd, ph=ph, ph_lengths=ph_lengths, mel_lengths=mel_lengths, ph2words=ph2words, train=False, global_steps=self.global_step)
                
                technique_target = sample['tech_ids']
                techs = sample['techs']
                _, T, binary_num_techs = output['tech_logits'].shape
                tech_acc = torch.zeros([binary_num_techs], device=output['tech_logits'].device)
                tech_num = torch.zeros([binary_num_techs], device=output['tech_logits'].device)
                
                for idx in range(bsz):
                    item_name = sample['item_name'][idx]
                    label = sample['ph'][idx].data.cpu()[:ph_lengths[idx]]
                    ph_durs = sample['ph_durs'][idx].data.cpu()[:ph_lengths[idx]]
                    ph2word = sample['ph2words'][idx].data.cpu()[:ph_lengths[idx]]
                    ph_of = output['ph_of_list'][idx]
                    pred_tier = onset_offset_to_point_tier(ph_of, hop_size_second=hparams['hop_size'] / hparams['audio_sample_rate'])

                    mintime = 0.
                    target_tier = PointTier(name='phone')
                    for mark, time in zip(label, ph_durs):
                        target_tier.add(mintime, mark)
                        mintime += time.item()
                    target_tier.add(mintime, 0)
                    
                    ignored = [0, 1]
                    ph_pred_tier = remove_ignored_phonemes(ignored, pred_tier)
                    ph_target_tier = remove_ignored_phonemes(ignored, target_tier)
                    
                    for key, metric in metrics.items():
                        try:
                            if key=="BoundaryEditRatio": 
                                metric.update(ph_pred_tier, ph_target_tier)
                            else:
                                metric.update(pred_tier, target_tier)
                        except AssertionError as e:
                            raise e
                    len_tech = ph_lengths[idx]
                    
                    tech_logits = output['tech_logits'][idx]
                    tech_pred = torch.sigmoid(tech_logits)  # [B, ph_length, tech_num]
                    tech_pred = (tech_pred > tech_threshold).long()
                    tech = techs[idx]
                    
                    batch_nonzero_indices = []
                    for tech_idx in range(binary_num_techs):
                        if torch.any(tech[:, tech_idx] > 0):
                            batch_nonzero_indices.append(tech_idx)
                            
                    group = technique_target[idx]
                    if group!=None:
                        valid_bin_tech_indices = [index for index in batch_nonzero_indices]
                        tech_acc[valid_bin_tech_indices] += (tech_pred[:len_tech, valid_bin_tech_indices] == tech[:len_tech, valid_bin_tech_indices]).sum(dim=0)
                        tech_num[valid_bin_tech_indices] += len_tech

                    note_bd_pred = note_bd_gt = sample['note_bd'][idx]
                    if self.global_step > hparams['vq_note_start']:
                        note_bd_pred = output['note_bd_pred'][idx]
                    note_pred = output['note_pred'][idx]
                    note_gt = sample['notes'][idx]
                    note_onset_f_1, note_offset_f_1, overlap_f_1, avg_overlap_ratio_1, rpa_1 = self.validate_note(note_bd_pred, note_bd_gt, note_pred, note_gt)
                    note_onset_f += note_onset_f_1
                    note_offset_f += note_offset_f_1
                    overlap_f += overlap_f_1
                    avg_overlap_ratio += avg_overlap_ratio_1
                    rpa += rpa_1
                    sample_num += 1
                
                outputs['losses']['note_onset_f'] = note_onset_f / sample_num
                outputs['losses']['note_offset_f'] = note_offset_f / sample_num
                outputs['losses']['overlap_f'] = overlap_f / sample_num
                outputs['losses']['avg_overlap_ratio'] = avg_overlap_ratio / sample_num
                outputs['losses']['rpa'] = rpa / sample_num

                tech_acc = tech_acc / tech_num
                tech_acc[tech_num == 0] = 1
                outputs['losses']['bubble_acc'] = tech_acc[0] 
                outputs['losses']['breathe_acc'] = tech_acc[1]
                outputs['losses']['pharyngeal_acc'] = tech_acc[2]
                outputs['losses']['vibrato_acc'] = tech_acc[3]
                outputs['losses']['glissando_acc'] = tech_acc[4]
                outputs['losses']['mixed_acc'] = tech_acc[5]
                outputs['losses']['falsetto_acc'] = tech_acc[6]
                outputs['losses']['weak_acc'] = tech_acc[7]
                outputs['losses']['strong_acc'] = tech_acc[8]
                                    
                for key, metric in metrics.items():
                    if key=="IntersectionOverUnion":
                        outputs['losses'][key] = metric.compute(add_all=True)
                    else:
                        outputs['losses'][key] = metric.compute()
                
                language_target = sample['languages']
                gender_target = sample['genders']
                emotion_target = sample['emotions']
                method_target = sample['singing_methods']
                pace_target = sample['paces']
                range_target = sample['ranges']
                
                technique_pred = torch.argmax(output['technique_logits'], dim=-1)
                language_pred = torch.argmax(output['language_logits'], dim=-1)
                gender_pred = torch.argmax(output['gender_logits'], dim=-1)
                emotion_pred = torch.argmax(output['emotion_logits'], dim=-1)
                method_pred = torch.argmax(output['method_logits'], dim=-1)
                pace_pred = torch.argmax(output['pace_logits'], dim=-1)
                range_pred = torch.argmax(output['range_logits'], dim=-1)
                
                def calculate_accuracy(pred, target):
                    valid_indices = [i for i, val in enumerate(target) if val is not None]
                    if len(valid_indices) == 0:
                        return 1.0
                    valid_pred = torch.tensor([pred[i].item() for i in valid_indices], device=pred.device)
                    valid_target = torch.tensor([target[i] for i in valid_indices], device=pred.device)
                    correct = (valid_pred == valid_target).sum().item()
                    accuracy = correct / len(valid_indices)
                    return accuracy
                
                technique_accuracy = calculate_accuracy(technique_pred, technique_target)
                language_accuracy = calculate_accuracy(language_pred, language_target)
                gender_accuracy = calculate_accuracy(gender_pred, gender_target)
                emotion_accuracy = calculate_accuracy(emotion_pred, emotion_target)
                method_accuracy = calculate_accuracy(method_pred, method_target)
                pace_accuracy = calculate_accuracy(pace_pred, pace_target)
                range_accuracy = calculate_accuracy(range_pred, range_target)
                
                outputs['losses']['technique_accuracy'] = technique_accuracy
                outputs['losses']['language_accuracy'] = language_accuracy
                outputs['losses']['gender_accuracy'] = gender_accuracy
                outputs['losses']['emotion_accuracy'] = emotion_accuracy
                outputs['losses']['method_accuracy'] = method_accuracy
                outputs['losses']['pace_accuracy'] = pace_accuracy
                outputs['losses']['range_accuracy'] = range_accuracy
                
            self.save_valid_result(sample, batch_idx, model_out)
        outputs = tensors_to_scalars(outputs)
        return outputs
    
    def validation_end(self, outputs):
        return super(StarsTask, self).validation_end(outputs)

    def save_valid_result(self, sample, batch_idx, model_out):
        pass

    def test_start(self):
        self.metrics = {
                    "BoundaryEditRatio": BoundaryEditRatio(),
                    "VlabelerEditRatio10ms": VlabelerEditRatio(move_tolerance=0.01),
                    "VlabelerEditRatio20ms": VlabelerEditRatio(move_tolerance=0.02),
                    "VlabelerEditRatio50ms": VlabelerEditRatio(move_tolerance=0.05),
                    "IntersectionOverUnion": IntersectionOverUnion(),
                }
        
        self.stylemetrics = {
                    "technique": StyleAcc(),
                    "language": StyleAcc(),
                    "gender": StyleAcc(),
                    "emotion": StyleAcc(),
                    "method": StyleAcc(),
                    "pace": StyleAcc(),
                    "range": StyleAcc(),
                }
        self.gen_dir = os.path.join(
            hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        self.num_techs = 9
        self.tech_acc = torch.zeros([self.num_techs])
        self.tech_num = torch.zeros([self.num_techs])
        self.tech_tp = torch.zeros(self.num_techs)
        self.tech_fp = torch.zeros(self.num_techs)
        self.tech_fn = torch.zeros(self.num_techs)
        self.tech_threshold = 0.5
        os.makedirs(self.gen_dir, exist_ok=True)
        os.makedirs(f'{self.gen_dir}/textgrid', exist_ok=True)
        os.makedirs(f'{self.gen_dir}/midi', exist_ok=True)
        self.note_onset_f, self.note_offset_f, self.overlap_f, self.avg_overlap_ratio, self.rpa = 0, 0, 0, 0, 0
        self.sample_num = 0
        self.COn_scores = np.zeros(3)
        self.COnP_scores = np.zeros(4)
        self.COnPOff_scores = np.zeros(4)
        self.melody_scores = np.zeros(5)

    def test_step(self, sample, batch_idx):
        _, output = self.run_model(sample, infer=True)
        bsz = sample['nsamples']
        ph_lengths = sample['ph_lengths']
        mel_lengths = sample['mel_lengths']
        techs = sample['techs']
        technique_target = sample['tech_ids']
        _, T, binary_num_techs = output['tech_logits'].shape
        self.tech_acc = self.tech_acc.to(output['tech_logits'].device)
        self.tech_num = self.tech_num.to(output['tech_logits'].device)
        self.tech_tp = self.tech_tp.to(output['tech_logits'].device)
        self.tech_fp = self.tech_fp.to(output['tech_logits'].device)
        self.tech_fn = self.tech_fn.to(output['tech_logits'].device)
        for idx in range(bsz):
            item_name = sample['item_name'][idx]
            label = sample['ph'][idx].data.cpu()[:ph_lengths[idx]]
            ph_durs = sample['ph_durs'][idx].data.cpu()[:ph_lengths[idx]]
            mel = sample['mels'][idx].data.cpu()[:mel_lengths[idx]]
            f0 = denorm_f0(sample['f0'], sample['uv'])[idx].cpu().numpy()[:mel_lengths[idx]]
            word = sample['words'][idx]

            ph_of = output['ph_of_list'][idx]
            word_of = output['word_of_list'][idx]
            ph_len = len(ph_of)
            dp_matrix = output['dp_matrix_list'][idx]
            get_textgrid(
                mel_lengths[idx], ph_of, word_of, 
                tech_pred=None, 
                draw=True, 
                hop_size_second=hparams['hop_size'] / hparams['audio_sample_rate'], 
                id2token=self.ph_encoder.id_to_token, 
                word_list=word, 
                mel=mel, 
                dp_matrix=dp_matrix, 
                tg_fn=os.path.join(self.gen_dir, 'textgrid', f'{item_name}.TextGrid')
            )
            
            pred_tier = onset_offset_to_point_tier(ph_of, hop_size_second=hparams['hop_size'] / hparams['audio_sample_rate'])
            
            mintime = 0.
            target_tier = PointTier(name='phone')
            for mark, time in zip(label, ph_durs):
                target_tier.add(mintime, mark)
                mintime += time.item()
            target_tier.add(mintime, 0)
            
            ignored = [0, 1]
            ph_pred_tier = remove_ignored_phonemes(ignored, pred_tier)
            ph_target_tier = remove_ignored_phonemes(ignored, target_tier)
            
            for key, metric in self.metrics.items():
                try:
                    if key=="BoundaryEditRatio": 
                        metric.update(ph_pred_tier, ph_target_tier)
                    else:
                        metric.update(pred_tier, target_tier)
                except AssertionError as e:
                    raise e
            tech_logits = output['tech_logits'][idx]
            tech_pred = torch.sigmoid(tech_logits)  # [B, ph_length, tech_num]
            tech_pred = (tech_pred > self.tech_threshold).long()
            tech = techs[idx]
            len_tech = ph_lengths[idx]
            batch_nonzero_indices = []
            for tech_idx in range(binary_num_techs):
                if torch.any(tech[:, tech_idx] > 0):
                    batch_nonzero_indices.append(tech_idx)
            group = technique_target[idx]
            if group!=None:
                valid_bin_tech_indices = [index for index in batch_nonzero_indices if index < binary_num_techs]
                self.tech_acc[valid_bin_tech_indices] += (tech_pred[:len_tech, valid_bin_tech_indices] == tech[:len_tech, valid_bin_tech_indices]).sum(dim=0)
                self.tech_num[valid_bin_tech_indices] += len_tech
                tech_true = tech[:len_tech, :].long()
                tech_pred_bin = tech_pred[:len_tech, :]
                for tech_idx in valid_bin_tech_indices:
                    pred = tech_pred_bin[:, tech_idx]
                    true = tech_true[:, tech_idx]
                    tp = (pred & true).sum()
                    fp = (pred & ~true).sum()
                    fn = (~pred & true).sum()
                    tech_precision = tp / (tp + fp + 1e-8)
                    tech_recall = tp / (tp + fn + 1e-8)
                    tech_f1 = 2 * (tech_precision * tech_recall) / (tech_precision + tech_recall + 1e-8)
                    self.tech_tp[tech_idx] += tp
                    self.tech_fp[tech_idx] += fp
                    self.tech_fn[tech_idx] += fn
            note_bd_pred = note_bd_gt = sample['note_bd'][idx]
            if self.global_step > hparams['vq_note_start']:
                note_bd_pred = output['note_bd_pred'][idx]
            note_pred = output['note_pred'][idx]
            note_gt = sample['notes'][idx]
            
            note_onset_f, note_offset_f, overlap_f, avg_overlap_ratio, rpa = self.validate_note(note_bd_pred, note_bd_gt, note_pred, note_gt)
            self.note_onset_f += note_onset_f
            self.note_offset_f += note_offset_f
            self.overlap_f += overlap_f
            self.avg_overlap_ratio += avg_overlap_ratio
            self.rpa += rpa

            base_fn = f'{item_name}[%s]'.replace(' ', '_')
            note_durs_gt = sample['note_durs'][idx].cpu().numpy()
            word_bd = sample['word_bd'][idx].cpu().numpy()
            word_durs = sample['word_durs'][idx].cpu().numpy()
            note_bd_prob = torch.sigmoid(output['note_bd_logits'])[idx].cpu().numpy()[:mel_lengths[idx]]
            self.save_note(base_fn, f0, note_bd_prob, note_bd_pred, note_bd_gt, note_pred, note_gt, word_bd, word_durs, note_durs_gt)
            
            self.sample_num += 1

        technique_target = sample['tech_ids']
        language_target = sample['languages']
        gender_target = sample['genders']
        emotion_target = sample['emotions']
        method_target = sample['singing_methods']
        pace_target = sample['paces']
        range_target = sample['ranges']
        
        technique_pred = torch.argmax(output['technique_logits'], dim=-1)
        language_pred = torch.argmax(output['language_logits'], dim=-1)
        gender_pred = torch.argmax(output['gender_logits'], dim=-1)
        emotion_pred = torch.argmax(output['emotion_logits'], dim=-1)
        method_pred = torch.argmax(output['method_logits'], dim=-1)
        pace_pred = torch.argmax(output['pace_logits'], dim=-1)
        range_pred = torch.argmax(output['range_logits'], dim=-1)
        
        self.stylemetrics['technique'].update(technique_pred, technique_target)
        self.stylemetrics['language'].update(language_pred, language_target)
        self.stylemetrics['gender'].update(gender_pred, gender_target)
        self.stylemetrics['emotion'].update(emotion_pred, emotion_target)
        self.stylemetrics['method'].update(method_pred, method_target)
        self.stylemetrics['pace'].update(pace_pred, pace_target)
        self.stylemetrics['range'].update(range_pred, range_target)        
        
        return {}


    def test_end(self, outputs):
        for key, metric in self.metrics.items():
            if key=="IntersectionOverUnion":
                value = metric.compute(add_all=True)
            else:
                value = metric.compute()
            print(f'|   {key}   |   {value:.3f}')   
            
        for key, metric in self.stylemetrics.items():
            value = metric.compute()
            print(f'|   {key}   |   {value:.3f}')   

        tech_acc = self.tech_acc / self.tech_num
        tech_precision = self.tech_tp / (self.tech_tp + self.tech_fp + 1e-8)
        tech_recall = self.tech_tp / (self.tech_tp + self.tech_fn + 1e-8)
        tech_f1 = 2 * (tech_precision * tech_recall) / (tech_precision + tech_recall + 1e-8)
        print('                         Acc         f1          precision       recall       ')
        print(f'|   bubble      |   {tech_acc[0]:.3f}|   {tech_f1[0].item():.3f}|   {tech_precision[0].item():.3f}|   {tech_recall[0].item():.3f}')
        print(f'|   breathe     |   {tech_acc[1]:.3f}|   {tech_f1[1].item():.3f}|   {tech_precision[1].item():.3f}|   {tech_recall[1].item():.3f}')
        print(f'|   pharyngeal  |   {tech_acc[2]:.3f}|   {tech_f1[2].item():.3f}|   {tech_precision[2].item():.3f}|   {tech_recall[2].item():.3f}')
        print(f'|   vibrato     |   {tech_acc[3]:.3f}|   {tech_f1[3].item():.3f}|   {tech_precision[3].item():.3f}|   {tech_recall[3].item():.3f}')
        print(f'|   glissando   |   {tech_acc[4]:.3f}|   {tech_f1[4].item():.3f}|   {tech_precision[4].item():.3f}|   {tech_recall[4].item():.3f}')
        print(f'|   mixed       |   {tech_acc[5]:.3f}|   {tech_f1[5].item():.3f}|   {tech_precision[5].item():.3f}|   {tech_recall[5].item():.3f}')
        print(f'|   falsetto    |   {tech_acc[6]:.3f}|   {tech_f1[6].item():.3f}|   {tech_precision[6].item():.3f}|   {tech_recall[6].item():.3f}')
        print(f'|   weak        |   {tech_acc[7]:.3f}|   {tech_f1[7].item():.3f}|   {tech_precision[7].item():.3f}|   {tech_recall[7].item():.3f}')
        print(f'|   strong      |   {tech_acc[8]:.3f}|   {tech_f1[8].item():.3f}|   {tech_precision[8].item():.3f}|   {tech_recall[8].item():.3f}')
        print(f'|   all         |   {tech_acc.mean():.3f}|   {tech_f1.mean():.3f}|   {tech_precision.mean():.3f}|   {tech_recall.mean():.3f}')
        self.note_onset_f /= self.sample_num
        self.note_offset_f /= self.sample_num
        self.overlap_f /= self.sample_num
        self.avg_overlap_ratio /= self.sample_num
        self.rpa /= self.sample_num
        print(f'|   note_onset_f        |   {self.note_onset_f:.3f}')
        print(f'|   note_offset_f       |   {self.note_offset_f:.3f}')
        print(f'|   overlap_f           |   {self.overlap_f:.3f}')
        print(f'|   avg_overlap_ratio   |   {self.avg_overlap_ratio:.3f}')
        print(f'|   rpa                 |   {self.rpa:.3f}')

        return {}
    
    def save_note(self, base_fn, gt_f0, note_bd_prob, note_bd_pred, note_bd_gt, note_pred, note_gt, word_bd, word_durs, note_durs_gt):
        note_bd_pred = note_bd_pred.data.cpu().numpy()
        note_bd_gt = note_bd_gt.data.cpu().numpy()
        note_gt = note_gt.data.cpu().numpy()
        orig_note_pred = note_pred = note_pred.data.cpu().numpy()
        note_itv_pred = boundary2Interval(note_bd_pred)
        note_itv_gt = boundary2Interval(note_bd_gt)

        os.makedirs(f'{self.gen_dir}/midi', exist_ok=True)
        fn = base_fn % 'P'
        pred_fn = f'{self.gen_dir}/midi/{fn}.mid'
        note_itv_pred_secs, note2words = regulate_real_note_itv(note_itv_pred, note_bd_pred, word_bd, word_durs, hparams['hop_size'], hparams['audio_sample_rate'])
        note_pred, note_itv_pred_secs, note2words = regulate_ill_slur(note_pred, note_itv_pred_secs, note2words)
        save_midi(note_pred, note_itv_pred_secs, pred_fn)

        fn = base_fn % 'G'
        gt_fn = f'{self.gen_dir}/midi/{fn}.mid'
        note_itv_gt_secs = np.zeros((note_durs_gt.shape[0], 2))
        note_offsets = np.cumsum(note_durs_gt)
        for idx in range(len(note_offsets) - 1):
            note_itv_gt_secs[idx, 1] = note_itv_gt_secs[idx + 1, 0] = note_offsets[idx]
        note_itv_gt_secs[-1, 1] = note_offsets[-1]
        save_midi(note_gt, note_itv_gt_secs, gt_fn)

        fn = base_fn % 'V'
        fig = plt.figure()
        plt.plot(gt_f0, color='blue', label='gt f0', lw=1)
        midi_pred = np.zeros(note_bd_pred.shape[0])
        for i, itv in enumerate(np.round(note_itv_pred).astype(int)):
            midi_pred[itv[0]: itv[1]] = orig_note_pred[i]
        midi_pred = midi_to_hz(midi_pred)
        plt.plot(midi_pred, color='green', label='pred midi')
        midi_gt = np.zeros(note_bd_gt.shape[0])
        for i, itv in enumerate(np.round(note_itv_gt).astype(int)):
            midi_gt[itv[0]: itv[1]] = note_gt[i]
        midi_gt = midi_to_hz(midi_gt)
        plt.plot(midi_gt, color='red', label='gt midi', lw=1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.gen_dir}/midi/{fn}.png', format='png')
        plt.close(fig)

        return None

    def validate_note(self, note_bd_pred, note_bd_gt, note_pred, note_gt):
        note_itv_gt = boundary2Interval(note_bd_gt.data.cpu().numpy()) * hparams['hop_size'] / hparams['audio_sample_rate']
        note_itv_pred = boundary2Interval(note_bd_pred.data.cpu().numpy()) * hparams['hop_size'] / hparams['audio_sample_rate']
        note_gt = midi_to_hz(note_gt.data.cpu().numpy())
        note_pred = midi_to_hz(note_pred.data.cpu().numpy())
        note_gt, note_itv_gt = validate_pitch_and_itv(note_gt, note_itv_gt)
        note_pred, note_itv_pred = validate_pitch_and_itv(note_pred, note_itv_pred)
        try:
            note_onset_p, note_onset_r, note_onset_f = mir_eval.transcription.onset_precision_recall_f1(
                note_itv_gt, note_itv_pred, onset_tolerance=0.05, strict=False, beta=1.0)
            note_offset_p, note_offset_r, note_offset_f = mir_eval.transcription.offset_precision_recall_f1(
                note_itv_gt, note_itv_pred, offset_min_tolerance=0.05, strict=False, beta=1.0)

            overlap_p, overlap_r, overlap_f, avg_overlap_ratio = mir_eval.transcription.precision_recall_f1_overlap(
                note_itv_gt, note_gt, note_itv_pred, note_pred, onset_tolerance=0.05, pitch_tolerance=50.0,
                offset_ratio=0.2, offset_min_tolerance=0.05, strict=False, beta=1.0)
            vr, vfa, rpa, rca, oa = melody_eval_pitch_and_itv(
                note_gt, note_itv_gt, note_pred, note_itv_pred, hparams['hop_size'], hparams['audio_sample_rate'])
        except Exception as err:
            note_onset_p, note_onset_r, note_onset_f = 0, 0, 0
            note_offset_p, note_offset_r, note_offset_f = 0, 0, 0
            overlap_p, overlap_r, overlap_f, avg_overlap_ratio = 0, 0, 0, 0
            vr, vfa, rpa, rca, oa = 0, 0, 0, 0, 0
            _, exc_value, exc_tb = sys.exc_info()
            tb = traceback.extract_tb(exc_tb)[-1]
            print(f'{err}: {exc_value} in {tb[0]}:{tb[1]} "{tb[2]}" in {tb[3]}')
        
        return note_onset_f, note_offset_f, overlap_f, avg_overlap_ratio, rpa
