import os
import sys
import time
import yaml
import shutil
import random
import argparse
import hashlib
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader

from dataloader import REMISkylineToMidiTransformerDataset
from utils import pickle_load

sys.path.append('./model')
train_steps = 0
# SS26 Stage 2 PAD diagnostic: {} keeps the model default (PAD ignored);
# {'ignore_index': -100} reproduces the historical objective that counted PAD.
PAD_LOSS_KWARGS = {}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_epoch(log_file, log_data, is_init=False):
    if is_init:
        with open(log_file, 'w') as f:
            f.write('{:4} {:8} {:12} {:12}\n'.format('ep', 'steps', 'recons_loss', 'ep_time'))

    with open(log_file, 'a') as f:
        f.write('{:<4} {:<8} {:<12} {:<12}\n'.format(
            log_data['ep'], log_data['steps'], round(log_data['recons_loss'], 5), round(log_data['time'], 2)
        ))


def train_model(epoch, model, dloader, optim, sched, pad_token, model_type="performer"):
    global train_steps
    model.train()
    recons_loss_rec = 0.
    accum_samples = 0

    print('[epoch {:03d}] training ...'.format(epoch))
    print('[epoch {:03d}] # batches = {}'.format(epoch, len(dloader)))
    st = time.time()

    optim.zero_grad()
    for batch_idx, batch_samples in enumerate(dloader):
        if max_train_steps is not None and train_steps >= max_train_steps:
            break

        batch_dec_inp = batch_samples['dec_input'].cuda(gpuid)
        batch_dec_tgt = batch_samples['dec_target'].cuda(gpuid)
        batch_track_mask = batch_samples['track_mask'].cuda(gpuid)
        batch_inp_lens = batch_samples['length']

        batch_chord_idx = batch_samples['chord_idx'].cuda(gpuid)
        batch_melody_idx = batch_samples['melody_idx'].cuda(gpuid)

        train_steps += 1

        # get logits from model
        if model_type == "performer":
            omit_feature_map_draw = random.random() > redraw_prob
            dec_logits = model(
                batch_dec_inp,
                seg_inp=batch_track_mask,
                chord_inp=None,
                attn_kwargs={'omit_feature_map_draw': omit_feature_map_draw}
            )
        elif model_type == "gpt2":
            omit_feature_map_draw = True  # dummy var here
            dec_logits = model(
                batch_dec_inp,
                seg_inp=batch_track_mask,
                chord_inp=None,
            )
        losses = model.compute_loss(dec_logits, batch_dec_tgt, **PAD_LOSS_KWARGS)

        # clip gradient & update model
        if accum_steps > 1:
            losses['total_loss'] /= accum_steps
        losses['total_loss'].backward()

        batch_size = batch_samples['id'].size(0)
        recons_loss_rec += batch_size * losses['recons_loss'].item()
        accum_samples += batch_size

        is_update_step = (train_steps % accum_steps) == 0 or (batch_idx + 1) == len(dloader)
        if is_update_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()
            optim.zero_grad()

            total_acc, chord_acc, melody_acc, others_acc = \
                compute_accuracy(dec_logits.cpu(), batch_dec_tgt.cpu(), batch_chord_idx, batch_melody_idx, pad_token)

            print(
                ' -- epoch {:03d} | batch {:03d}/{:03d}: len: {}\n   '
                '* loss = {:.4f}, total_acc = {:.4f}, chord_acc = {:.4f}, '
                'melody_acc = {:.4f}, others_acc = {:.4f}, '
                'step = {}, time_elapsed = {:.2f} secs | redraw: {}'.format(
                    epoch, batch_idx + 1, len(dloader), batch_inp_lens,
                           recons_loss_rec / accum_samples, total_acc,
                    chord_acc, melody_acc, others_acc, train_steps,
                           time.time() - st, (not omit_feature_map_draw)
                ))

            # anneal learning rate after each optimizer update
            update_steps = (train_steps + accum_steps - 1) // accum_steps
            if update_steps < warmup_steps:
                curr_lr = max_lr * update_steps / warmup_steps
                optim.param_groups[0]['lr'] = curr_lr
            else:
                sched.step(update_steps - warmup_steps)

        if not train_steps % log_interval:
            log_data = {
                'ep': epoch,
                'steps': train_steps,
                'recons_loss': recons_loss_rec / accum_samples,
                'time': time.time() - st
            }
            log_epoch(
                os.path.join(ckpt_dir, 'log.txt'), log_data,
                is_init=not os.path.exists(os.path.join(ckpt_dir, 'log.txt'))
            )

    print('[epoch {:03d}] training completed\n  -- loss = {:.4f}\n  -- time elapsed = {:.2f} secs.'.format(
        epoch, recons_loss_rec / accum_samples, time.time() - st
    ))
    log_data = {
        'ep': epoch,
        'steps': train_steps,
        'recons_loss': recons_loss_rec / accum_samples,
        'time': time.time() - st
    }
    log_epoch(
        os.path.join(ckpt_dir, 'log.txt'), log_data, is_init=not os.path.exists(os.path.join(ckpt_dir, 'log.txt'))
    )

    return recons_loss_rec / accum_samples


def validate(model, dloader, pad_token, rounds=1, model_type="performer"):
    model.eval()
    loss_rec = []
    total_acc_rec = []
    chord_acc_rec = []
    melody_acc_rec = []
    others_acc_rec = []
    nonpad_nll_sum = 0.0
    nonpad_tokens = 0

    with torch.no_grad():
        for r in range(rounds):
            print(' >> validating ... (round {})'.format(r + 1))
            for batch_idx, batch_samples in enumerate(dloader):
                batch_dec_inp = batch_samples['dec_input'].cuda(gpuid)
                batch_dec_tgt = batch_samples['dec_target'].cuda(gpuid)
                batch_track_mask = batch_samples['track_mask'].cuda(gpuid)
                batch_chord_idx = batch_samples['chord_idx'].cuda(gpuid)
                batch_melody_idx = batch_samples['melody_idx'].cuda(gpuid)

                if model_type == "performer":
                    omit_feature_map_draw = random.random() > redraw_prob
                    dec_logits = model(
                        batch_dec_inp,
                        seg_inp=batch_track_mask,
                        chord_inp=None,
                        attn_kwargs={'omit_feature_map_draw': omit_feature_map_draw}
                    )
                elif model_type == "gpt2":
                    dec_logits = model(
                        batch_dec_inp,
                        seg_inp=batch_track_mask,
                        chord_inp=None,
                    )

                losses = model.compute_loss(dec_logits, batch_dec_tgt, **PAD_LOSS_KWARGS)
                flat_tgt = batch_dec_tgt.contiguous().view(-1).long()
                nonpad_nll_sum += torch.nn.functional.cross_entropy(
                    dec_logits.view(-1, dec_logits.size(-1)), flat_tgt,
                    ignore_index=pad_token, reduction='sum').item()
                nonpad_tokens += int((flat_tgt != pad_token).sum().item())
                if not batch_idx % 5:
                    print('  -- batch #{:03d}: valloss = {:.4f}'.format(
                        batch_idx + 1, losses['recons_loss'].item()
                    ))
                loss_rec.append(losses['recons_loss'].item())

                total_acc, chord_acc, melody_acc, others_acc = \
                    compute_accuracy(dec_logits.cpu(), batch_dec_tgt.cpu(), batch_chord_idx, batch_melody_idx, pad_token)
                total_acc_rec.append(total_acc)
                chord_acc_rec.append(chord_acc)
                melody_acc_rec.append(melody_acc)
                others_acc_rec.append(others_acc)

    nonpad_ce = nonpad_nll_sum / max(nonpad_tokens, 1)
    return loss_rec, total_acc_rec, chord_acc_rec, melody_acc_rec, others_acc_rec, nonpad_ce


def compute_accuracy(dec_logits, dec_target, inp_chord, inp_melody, pad_token):
    # 确保所有张量在同一个设备（GPU）
    device = dec_logits.device
    dec_pred = dec_logits.argmax(-1).to(device)
    dec_target = dec_target.to(device)
    inp_chord = inp_chord.to(device)
    inp_melody = inp_melody.to(device)
    
    dec_pred = torch.argmax(dec_logits, dim=-1)
    dec_target = dec_target
    total_acc = np.mean(np.array((dec_pred[dec_target != pad_token] == dec_target[dec_target != pad_token])))
    chord_acc = np.mean(np.array((dec_pred[inp_chord == 1] == dec_target[inp_chord == 1])))
    melody_acc = np.mean(np.array((dec_pred[inp_melody == 1] == dec_target[inp_melody == 1])))
    others_acc = (total_acc * len(dec_target[dec_target != pad_token]) - chord_acc * len(dec_target[inp_chord == 1]) -
                  melody_acc * len(dec_target[inp_melody == 1])) / \
                 (len(dec_target[dec_target != pad_token]) - len(dec_target[inp_chord == 1]) - len(dec_target[inp_melody == 1]))
    return total_acc, chord_acc, melody_acc, others_acc


if __name__ == "__main__":
    # configuration
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-m', '--model_type',
                          choices=['performer', 'gpt2'],
                          help='model backbone', required=True)
    required.add_argument('-c', '--configuration',
                          choices=['stage2_melody/config/pop1k7_pretrain.yaml',
                                   'stage2_melody/config/pop1k7_pretrain_gpt2.yaml',
                                   'stage2_melody/config/pop1k7_pretrain_new_gpt2.yaml',
                                   'stage2_melody/config/hooktheory_true_melody_pretrain_v90_gpt2.yaml',
                                   'stage2_melody/config/hooktheory_true_melody_pretrain_v92_gpt2.yaml',
                                   'stage2_melody/config/emopia_finetune.yaml',
                                   'stage2_melody/config/emopia_finetune_gpt2.yaml',
                                   'stage2_melody/config/emopia_finetune_new_gpt2.yaml',
                                   'stage2_melody/config/emopia_true_melody_finetune_v90_gpt2.yaml',
                                   'stage2_melody/config/emopia_true_melody_finetune_v92_gpt2.yaml'],
                          help='configurations of training', required=True)
    required.add_argument('-r', '--representation',
                          choices=['remi', 'functional'],
                          help='representation for symbolic music', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_steps', type=int, default=None,
                        help='stop after this many training steps and save a smoke checkpoint')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='override the configured epoch limit')
    parser.add_argument('--patience', type=int, default=None,
                        help='override early-stopping patience')
    parser.add_argument('--ckpt_dir', default=None,
                        help='override the configured checkpoint directory')
    parser.add_argument('--pretrained_param_path', default=None,
                        help='override the configured pretrained parameter path')
    parser.add_argument('--pretrained_optim_path', default=None,
                        help='override the configured pretrained optimizer path')
    parser.add_argument('--reset_optimizer', action='store_true',
                        help='start a fresh optimizer when transferring pretrained parameters')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pad-loss', dest='pad_loss', choices=['ignore', 'count'], default='ignore',
                        help='ignore: PAD targets excluded (current default); count: historical objective')
    parser.add_argument('--select-metric', dest='select_metric', choices=['own', 'nonpad_ce'], default='own',
                        help='checkpoint selection: own validation loss (default) or shared non-PAD CE')
    parser.add_argument('--fixed-validation-start', type=int, default=None,
                        help='use the same admissible start bar for all validation pieces')
    args = parser.parse_args()

    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max_steps must be positive')
    if args.max_epochs is not None and args.max_epochs < 1:
        parser.error('--max_epochs must be positive')
    if args.patience is not None and args.patience < 1:
        parser.error('--patience must be positive')
    set_seed(args.seed)
    PAD_LOSS_KWARGS = {'ignore_index': -100} if args.pad_loss == 'count' else {}
    print('[pad-loss]', args.pad_loss, '[select-metric]', args.select_metric)
    max_train_steps = args.max_steps
    print('[seed]', args.seed, '[max_steps]', max_train_steps,
          '[max_epochs]', args.max_epochs)

    model_type = args.model_type
    if model_type == "performer":
        from model.music_performer import MusicPerformer

        model_klass = MusicPerformer
    elif model_type == "gpt2":
        from model.music_gpt2 import ImprovedMusicGPT2

        model_klass = ImprovedMusicGPT2
    else:
        raise NotImplementedError("Unsuppported model:", model_type)

    train_conf_path = args.configuration
    train_conf = yaml.load(open(train_conf_path, 'r'), Loader=yaml.FullLoader)
    print(train_conf)
    representation = args.representation

    # training configurations
    train_conf_ = train_conf['training']
    gpuid = train_conf_['gpuid']
    torch.cuda.set_device(gpuid)
    warmup_steps = train_conf_['warmup_steps']
    max_lr = train_conf_['lr']
    min_lr = train_conf_['lr_scheduler']['eta_min']
    lr_decay_steps = train_conf_['lr_scheduler']['T_max']
    redraw_prob = train_conf_.get('feat_redraw_prob', 0.0)
    max_epochs = args.max_epochs or train_conf_['num_epochs']
    ckpt_dir = args.ckpt_dir or train_conf_['ckpt_dir'].format(representation)
    pretrained_param_path = args.pretrained_param_path or train_conf_['trained_params']
    pretrained_optimizer_path = None if args.reset_optimizer else (
        args.pretrained_optim_path or train_conf_['trained_optim']
    )
    ckpt_interval = train_conf_['ckpt_interval']
    val_interval = 1
    log_interval = train_conf_['log_interval']
    accum_steps = train_conf_.get("accum_steps", 1)

    # dataloader configurations
    batch_size = train_conf['data_loader']['batch_size']
    train_split = train_conf['data_loader']['train_split']
    val_split = train_conf['data_loader']['val_split']
    data_path = train_conf['data_loader']['data_path'].format(representation)
    vocab_path = train_conf['data_loader']['vocab_path'].format(representation)
    event2idx = pickle_load(vocab_path)[0]
    idx2event = pickle_load(vocab_path)[1]

    # model configurations
    model_conf = train_conf['model']
    predict_key = False

    # load dataset
    dset = REMISkylineToMidiTransformerDataset(
        data_dir=data_path,
        vocab_file=vocab_path,
        model_dec_seqlen=model_conf['max_len'],
        pieces=pickle_load(train_split),
        pad_to_same=True,
        predict_key=predict_key
    )
    val_dset = REMISkylineToMidiTransformerDataset(
        data_dir=data_path,
        vocab_file=vocab_path,
        model_dec_seqlen=model_conf['max_len'],
        pieces=pickle_load(val_split),
        pad_to_same=True,
        predict_key=predict_key,
        appoint_st_bar=args.fixed_validation_start
    )
    print('[info] # training pieces:', len(dset.pieces))
    dloader = DataLoader(
        dset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed)
    )
    val_dloader = DataLoader(
        val_dset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers
    )

    # load model
    if model_type == "performer":
        model = model_klass(
            dset.vocab_size, model_conf['n_layer'], model_conf['n_head'],
            model_conf['d_model'], model_conf['d_ff'], model_conf['d_embed'],
            use_segment_emb=model_conf['use_segemb'], n_segment_types=2,
            favor_feature_dims=model_conf['feature_map']['n_dims'],
            use_chord_mhot_emb=False
        ).cuda(gpuid)
    elif model_type == "gpt2":
        model = model_klass(
            dset.vocab_size, model_conf['n_layer'], model_conf['n_head'],
            model_conf['d_model'], model_conf['d_ff'], model_conf['d_embed'],
            # use_segment_emb=model_conf['use_segemb'], n_segment_types=2,
            # use_chord_mhot_emb=False
        ).cuda(gpuid)

    if pretrained_param_path:
        pretrained_dict = torch.load(pretrained_param_path, map_location='cpu')
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if 'feature_map.omega' not in k
        }
        model_state_dict = model.state_dict()
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict)

    model.train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# params:', n_params)
    # print('segemb:', model.segemb)

    opt_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(opt_params, lr=max_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, lr_decay_steps, eta_min=min_lr
    )
    if pretrained_optimizer_path:
        optimizer.load_state_dict(
            torch.load(pretrained_optimizer_path, map_location='cpu')
        )

    params_dir = os.path.join(ckpt_dir, 'params/')
    optimizer_dir = os.path.join(ckpt_dir, 'optim/')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
    if not os.path.exists(optimizer_dir):
        os.makedirs(optimizer_dir)

    shutil.copy(train_conf_path, os.path.join(ckpt_dir, 'config.yaml'))
    import json as _json
    with open(os.path.join(ckpt_dir, 'training_objective.json'), 'w') as _fh:
        _state_hash = hashlib.sha256()
        for _name, _tensor in sorted(model.state_dict().items()):
            _state_hash.update(_name.encode())
            _state_hash.update(_tensor.detach().cpu().contiguous().numpy().tobytes())
        _json.dump({'pad_loss': args.pad_loss, 'select_metric': args.select_metric,
                    'pad_token': int(dset.pad_token), 'vocab_size': int(dset.vocab_size),
                    'loss_kwargs': PAD_LOSS_KWARGS, 'seed': args.seed,
                    'initial_state_sha256': _state_hash.hexdigest(),
                    'fixed_validation_start': args.fixed_validation_start}, _fh, indent=2)

    # 在训练开始前初始化早停相关变量
    best_val_loss = float('inf')
    no_improve_epochs = 0
    patience = args.patience or train_conf['training'].get('patience', 10)

    for ep in range(max_epochs):
        loss = train_model(ep + 1, model, dloader, optimizer, scheduler, dset.pad_token, model_type=model_type)
        if max_train_steps is not None and train_steps >= max_train_steps:
            torch.save(model.state_dict(), os.path.join(params_dir, 'smoke_params.pt'))
            torch.save(optimizer.state_dict(), os.path.join(optimizer_dir, 'smoke_optim.pt'))
            print('[smoke] reached {} steps; checkpoint saved'.format(train_steps))
            break
        if not (ep + 1) % ckpt_interval:
            torch.save(model.state_dict(),
                    os.path.join(params_dir, 'ep{:03d}_loss{:.3f}_params.pt'.format(ep + 1, loss)))
            torch.save(optimizer.state_dict(),
                    os.path.join(optimizer_dir, 'ep{:03d}_loss{:.3f}_optim.pt'.format(ep + 1, loss)))
        if not (ep + 1) % val_interval:
            val_losses, total_acc_rec, chord_acc_rec, melody_acc_rec, others_acc_rec, val_nonpad_ce = \
                validate(model, val_dloader, val_dset.pad_token, model_type=model_type)
            own_val_loss = np.mean(val_losses)
            current_val_loss = val_nonpad_ce if args.select_metric == 'nonpad_ce' else own_val_loss
            if not np.isfinite(current_val_loss):
                raise FloatingPointError('Non-finite checkpoint selection metric')
            with open(os.path.join(ckpt_dir, 'val_nonpad_ce.txt'), 'a') as f:
                f.write('ep{:03d}\town_valloss={:.6f}\tnonpad_ce={:.6f}\tselect={}\n'.format(
                    ep + 1, own_val_loss, val_nonpad_ce, args.select_metric))
            
            with open(os.path.join(ckpt_dir, 'valloss.txt'), 'a') as f:
                f.write("ep{:03d} | loss: {:.3f} | valloss: {:.3f} (±{:.3f}) | total_acc: {:.3f} | "
                        "chord_acc: {:.3f} | melody_acc: {:.3f} | others_acc: {:.3f}\n".format(
                    ep + 1, loss, own_val_loss, np.std(val_losses),
                    np.mean(total_acc_rec), np.mean(chord_acc_rec), np.mean(melody_acc_rec), np.mean(others_acc_rec)
                ))
            
            # 更新最佳验证损失和早停计数器
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                no_improve_epochs = 0
                # 保存最佳模型
                torch.save(model.state_dict(), os.path.join(params_dir, 'best_params.pt'))
                torch.save(optimizer.state_dict(), os.path.join(optimizer_dir, 'best_optim.pt'))
                print(f"Epoch {ep+1}: Validation loss improved to {best_val_loss:.3f}, saving best model...")
            else:
                no_improve_epochs += 1
                print(f"Epoch {ep+1}: Validation loss did not improve. No improvement for {no_improve_epochs}/{patience} epochs.")
            
            # 早停检查
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {ep+1} epochs!")
                break
