import os
import sys
import shutil
import time
import yaml
import random
import torch
import argparse
import numpy as np
from torch import optim
from torch.utils.data import DataLoader

from model.plain_transformer import PlainTransformer  # 导入自定义的Transformer模型
from dataloader import SkylineFullSongTransformerDataset  # 导入自定义的数据加载器
from utils import pickle_load  # 导入工具函数，用于加载pickle文件

sys.path.append('./model/')  # 将模型路径添加到系统路径中，方便导入


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(epoch, model, dloader, optim, sched, pad_token):
    """
    训练函数
    :param epoch: 当前训练的epoch
    :param model: 模型
    :param dloader: 数据加载器
    :param optim: 优化器
    :param sched: 学习率调度器
    :param pad_token: 填充token
    """
    global train_steps
    model.train()  # 将模型设置为训练模式
    recons_loss_rec = 0.  # 记录重构损失
    accum_samples = 0  # 累计样本数

    print('[epoch {:03d}] training ...'.format(epoch))
    st = time.time()  # 记录开始时间

    for batch_idx, batch_samples in enumerate(dloader):
        # if batch_idx > 4:
        #     break
        mems = tuple()  # 初始化记忆单元
        # print ('[debug] got batch samples')
        for segment in range(max(batch_samples['n_seg'])):  # 遍历每个segment
            if max_train_steps is not None and train_steps >= max_train_steps:
                return recons_loss_rec / accum_samples, time.time() - st
            # print ('[debug] segment:', segment)

            model.zero_grad()  # 清空梯度

            # 将输入数据转移到GPU上
            dec_input = batch_samples['dec_inp_{}'.format(segment)].permute(1, 0).cuda()
            dec_target = batch_samples['dec_tgt_{}'.format(segment)].permute(1, 0).cuda()
            dec_seg_len = batch_samples['dec_seg_len_{}'.format(segment)].cuda()

            inp_chord = batch_samples['inp_chord_{}'.format(segment)]
            # inp_melody = batch_samples['inp_melody_{}'.format(segment)]
            # print ('[debug]', dec_input.size(), dec_target.size(), dec_seg_len.size())
            train_steps += 1  # 训练步数加1

            # print ('[debug] prior to model forward(), train steps:', train_steps)
            dec_logits, mems = \
                model(
                    dec_input, mems, dec_seg_len=dec_seg_len
                )  # 前向传播，得到输出和记忆单元

            # print ('[debug] got model output')
            # 计算损失
            losses = model.compute_loss(
                dec_logits, dec_target
            )

            # 计算准确率
            # total_acc, chord_acc, melody_acc, others_acc = \
            total_acc, chord_acc, others_acc = \
                compute_accuracy(dec_logits.cpu(), dec_target.cpu(), inp_chord, pad_token)

            # 反向传播并更新模型参数
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 梯度裁剪
            optim.step()  # 更新模型参数
            recons_loss_rec += batch_samples['id'].size(0) * losses['ce_loss'].item()  # 累计损失
            accum_samples += batch_samples['id'].size(0)  # 累计样本数

            # 学习率预热
            if train_steps < warmup_steps:
                curr_lr = max_lr * train_steps / warmup_steps
                optim.param_groups[0]['lr'] = curr_lr
            else:
                sched.step(train_steps - warmup_steps)  # 更新学习率

            # 每隔log_interval步记录一次日志
            if not train_steps % log_interval:
                log_data = {
                    'ep': epoch,
                    'steps': train_steps,
                    'ce_loss': recons_loss_rec / accum_samples,
                    'total': total_acc,
                    'chord': chord_acc,
                    # 'melody': melody_acc,
                    'others': others_acc,
                    'time': time.time() - st
                }
                log_epoch(
                    os.path.join(ckpt_dir, log_file), log_data,
                    is_init=not os.path.exists(os.path.join(ckpt_dir, log_file))
                )

        # 打印当前batch的训练结果
        print('-- ep {:03d} | batch {:03d}: loss = {:.4f}, total_acc = {:.4f}, '
            #   'chord_acc = {:.4f}, melody_acc = {:.4f}, others_acc = {:.4f}, '
              'chord_acc = {:.4f}, others_acc = {:.4f}, '
              'step = {}, time_elapsed = {:.2f} secs'.format(
            epoch,
            batch_idx,
            recons_loss_rec / accum_samples,
            total_acc,
            chord_acc,
            # melody_acc,
            others_acc,
            train_steps,
            time.time() - st
        ))

    return recons_loss_rec / accum_samples, time.time() - st  # 返回平均损失和训练时间


def validate(epoch, model, dloader, pad_token, rounds=1):
    """
    验证函数
    :param epoch: 当前epoch
    :param model: 模型
    :param dloader: 数据加载器
    :param pad_token: 填充token
    :param rounds: 验证轮数
    """
    model.eval()  # 将模型设置为评估模式
    recons_loss_rec = []  # 记录验证损失
    total_acc_rec = []  # 记录总准确率
    chord_acc_rec = []  # 记录和弦准确率
    melody_acc_rec = []  # 记录旋律准确率
    others_acc_rec = []  # 记录其他准确率

    print('[epoch {:03d}] validating ...'.format(epoch))
    with torch.no_grad():  # 不计算梯度
        for r in range(rounds):  # 进行多轮验证
            for batch_idx, batch_samples in enumerate(dloader):
                # if batch_idx > 4:
                #     break
                mems = tuple()  # 初始化记忆单元
                for segment in range(max(batch_samples['n_seg'])):  # 遍历每个segment
                    dec_input = batch_samples['dec_inp_{}'.format(segment)].permute(1, 0).cuda()
                    dec_target = batch_samples['dec_tgt_{}'.format(segment)].permute(1, 0).cuda()
                    dec_seg_len = batch_samples['dec_seg_len_{}'.format(segment)].cuda()
                    inp_chord = batch_samples['inp_chord_{}'.format(segment)]
                    # inp_melody = batch_samples['inp_melody_{}'.format(segment)]

                    dec_logits, mems = \
                        model(
                            dec_input, mems, dec_seg_len=dec_seg_len
                        )  # 前向传播，得到输出和记忆单元
                    # 计算损失
                    losses = model.compute_loss(
                        dec_logits, dec_target
                    )

                    # 计算准确率
                    total_acc, chord_acc, others_acc = \
                        compute_accuracy(dec_logits.cpu(), dec_target.cpu(), inp_chord, pad_token)

                    # 每隔10个batch打印一次验证结果
                    if not (batch_idx + 1) % 10:
                        print('  valloss = {:.4f}, total_acc = {:.4f}, chord_acc = {:.4f}, '
                            #   'melody_acc = {:.4f}, '
                              'others_acc = {:.4f}, '.format(
                            round(losses['ce_loss'].item(), 3),
                            total_acc,
                            chord_acc,
                            # melody_acc,
                            others_acc))
                    # 记录损失和准确率
                    recons_loss_rec.append(losses['ce_loss'].item())
                    total_acc_rec.append(total_acc)
                    chord_acc_rec.append(chord_acc)
                    # melody_acc_rec.append(melody_acc)
                    others_acc_rec.append(others_acc)

    # return recons_loss_rec, total_acc_rec, chord_acc_rec, melody_acc_rec, others_acc_rec  # 返回验证结果
    return recons_loss_rec, total_acc_rec, chord_acc_rec, others_acc_rec  


def log_epoch(log_file, log_data, is_init=False):
    """
    记录日志函数
    :param log_file: 日志文件路径
    :param log_data: 日志数据
    :param is_init: 是否是初始化日志文件
    """
    if is_init:  # 如果是初始化日志文件，写入表头
        with open(log_file, 'w') as f:
            f.write('{:4} {:8} {:12} {:12} {:12}\n'.format(
                'ep', 'steps', 'ce_loss', 'ep_time', 'total_time'
            ))

    # 追加日志数据
    with open(log_file, 'a') as f:
        f.write('{:<4} {:<8} {:<12} {:<12} {:<12}\n'.format(
            log_data['ep'],
            log_data['steps'],
            round(log_data['ce_loss'], 5),
            round(log_data['time'], 2),
            round(time.time() - init_time, 2)
        ))

    return


# def compute_accuracy(dec_logits, dec_target, inp_chord, inp_melody, pad_token):
def compute_accuracy(dec_logits, dec_target, inp_chord, pad_token):
    """
    计算准确率函数
    :param dec_logits: 模型输出的logits
    :param dec_target: 目标输出
    :param inp_chord: 输入的和弦
    :param inp_melody: 输入的旋律
    :param pad_token: 填充token
    """
    dec_pred = torch.argmax(dec_logits, dim=-1).permute(1, 0)  # 获取预测结果
    dec_target = dec_target.permute(1, 0)  # 调整目标输出的维度
    # 计算总准确率
    total_acc = np.mean(np.array((dec_pred[dec_target != pad_token] == dec_target[dec_target != pad_token])))
    # 计算和弦准确率
    chord_acc = np.mean(np.array((dec_pred[inp_chord == 1] == dec_target[inp_chord == 1])))
    # # 计算旋律准确率
    # melody_acc = np.mean(np.array((dec_pred[inp_melody == 1] == dec_target[inp_melody == 1])))
    # 计算其他准确率
    others_acc = (total_acc * len(dec_target[dec_target != pad_token]) - chord_acc * len(dec_target[inp_chord == 1]) 
                #   - melody_acc * len(dec_target[inp_melody == 1])
                  ) / \
                 (len(dec_target[dec_target != pad_token]) - len(dec_target[inp_chord == 1]) 
                #   - len(dec_target[inp_melody == 1])
                  )
    # return total_acc, chord_acc, melody_acc, others_acc
    return total_acc, chord_acc, others_acc


if __name__ == '__main__':
    # 配置参数解析
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--configuration',
                          choices=['stage1_chord/config/hooktheory_pretrain.yaml',
                                   'stage1_chord/config/emopia_finetune.yaml',
                                   'stage1_chord/config/pop1k7_pretrain.yaml',
                                   'stage1_chord/config/emopia_finetune_full.yaml'],
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
    args = parser.parse_args()

    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max_steps must be positive')
    if args.max_epochs is not None and args.max_epochs < 1:
        parser.error('--max_epochs must be positive')
    if args.patience is not None and args.patience < 1:
        parser.error('--patience must be positive')
    set_seed(args.seed)
    max_train_steps = args.max_steps
    print('[seed]', args.seed, '[max_steps]', max_train_steps,
          '[max_epochs]', args.max_epochs)

    # 加载配置文件
    config_path = args.configuration
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    print(config)
    representation = args.representation
    ckpt_dir = args.ckpt_dir or config['output']['ckpt_dir'].format(representation)

    # 设置GPU设备
    torch.cuda.device(config['device'])
    train_steps = 0 if config['training']['trained_steps'] is None else config['training']['trained_steps']
    start_epoch = 0 if config['training']['trained_epochs'] is None else config['training']['trained_epochs']
    warmup_steps = config['training']['warmup_steps']
    log_interval = config['training']['log_interval']
    max_lr = config['training']['max_lr']
    log_file = 'log.txt' if start_epoch == 0 else 'log_from_ep{:03d}.txt'.format(start_epoch)

    optim_ckpt_path = None if args.reset_optimizer else (
        args.pretrained_optim_path or config['pretrained_optim_path']
    )
    param_ckpt_path = args.pretrained_param_path or config['pretrained_param_path']
    max_epochs = args.max_epochs or config['training']['max_epoch']

    init_time = time.time()  # 记录初始化时间

    # 设置参数和优化器的保存路径
    params_dir = os.path.join(ckpt_dir, 'params/') if start_epoch == 0 \
        else os.path.join(ckpt_dir, 'params_from_ep{:03d}/'.format(start_epoch))
    optimizer_dir = os.path.join(ckpt_dir, 'optim/') if start_epoch == 0 \
        else os.path.join(ckpt_dir, 'optim_from_ep{:03d}/'.format(start_epoch))

    # 加载训练数据集
    dset = SkylineFullSongTransformerDataset(
        config['data']['data_dir'].format(representation),
        config['data']['vocab_path'].format(representation),
        pieces=pickle_load(config['data']['train_split']),
        # do_augment=True if "lmd" not in config['data']['data_dir'] else False,
        do_augment=False,
        model_dec_seqlen=config['model']['decoder']['tgt_len'],
        max_n_seg=config['data']['max_n_seg'],
        # max_pitch=108, min_pitch=48,
        max_pitch=108, min_pitch=21,
        convert_dict_event=True
    )

    # 加载验证数据集
    val_dset = SkylineFullSongTransformerDataset(
        config['data']['data_dir'].format(representation),
        config['data']['vocab_path'].format(representation),
        pieces=pickle_load(config['data']['val_split']),
        do_augment=False,
        model_dec_seqlen=config['model']['decoder']['tgt_len'],
        max_n_seg=config['data']['max_n_seg'],
        # max_pitch=108, min_pitch=48,
        max_pitch=108, min_pitch=21,
        convert_dict_event=True
    )
    print('[dset lens]', len(dset), len(val_dset))

    # 创建数据加载器
    dloader = DataLoader(
        dset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=dset.collate_fn
    )
    val_dloader = DataLoader(
        val_dset,
        batch_size=4,
        num_workers=args.num_workers,
        collate_fn=val_dset.collate_fn
    )

    # 加载模型配置
    mconf = config['model']
    # torch.cuda.set_device(1)
    # print (torch.cuda.current_device())
    model = PlainTransformer(
        mconf['d_word_embed'],
        dset.vocab_size,
        mconf['decoder']['n_layer'],
        mconf['decoder']['n_head'],
        mconf['decoder']['d_model'],
        mconf['decoder']['d_ff'],
        mconf['decoder']['mem_len'],
        mconf['decoder']['tgt_len'],
        dec_dropout=mconf['decoder']['dropout'],
        pre_lnorm=mconf['pre_lnorm']
    ).cuda()  # 将模型转移到GPU上
    print('[info] # params:', sum(p.numel() for p in model.parameters() if p.requires_grad))  # 打印模型参数量

    # 设置优化器和学习率调度器
    opt_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(opt_params, lr=config['training']['max_lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['lr_decay_steps'],
        eta_min=config['training']['min_lr']
    )

    # 加载预训练的优化器和模型参数
    if optim_ckpt_path:
        optimizer.load_state_dict(
            torch.load(optim_ckpt_path, map_location=config['device'])
        )

    if param_ckpt_path:
        pretrained_dict = torch.load(param_ckpt_path, map_location=config['device'])
        model.load_state_dict(
            pretrained_dict
        )

    # 创建检查点目录
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
    if not os.path.exists(optimizer_dir):
        os.makedirs(optimizer_dir)

    shutil.copy(config_path, os.path.join(ckpt_dir, 'config.yaml'))  # 复制配置文件到检查点目录

    # 在训练开始前初始化早停相关变量
    best_val_loss = float('inf')
    no_improve_epochs = 0
    patience = args.patience or config['training'].get('patience', 10)

    # 开始训练
    for ep in range(start_epoch, max_epochs):
        recons_loss, ep_time = train(ep + 1, model, dloader, optimizer, scheduler, dset.pad_token)
        if max_train_steps is not None and train_steps >= max_train_steps:
            torch.save(model.state_dict(), os.path.join(params_dir, 'smoke_params.pt'))
            torch.save(optimizer.state_dict(), os.path.join(optimizer_dir, 'smoke_optim.pt'))
            print('[smoke] reached {} steps; checkpoint saved'.format(train_steps))
            break
        # 每隔ckpt_interval个epoch保存一次模型参数和优化器状态
        if not (ep + 1) % config['output']['ckpt_interval']:
            torch.save(model.state_dict(),
                    os.path.join(params_dir, 'ep{:03d}_loss{:.3f}_params.pt'.format(ep + 1, recons_loss))
                    )
            torch.save(optimizer.state_dict(),
                    os.path.join(optimizer_dir, 'ep{:03d}_loss{:.3f}_optim.pt'.format(ep + 1, recons_loss))
                    )

        # 每隔val_interval个epoch进行一次验证
        if not (ep + 1) % config['training']['val_interval']:
            val_recons_losses, total_acc_rec, chord_acc_rec, others_acc_rec = \
                validate(ep + 1, model, val_dloader, val_dset.pad_token)
            current_val_loss = np.mean(val_recons_losses)
            
            # 更新最佳验证损失和早停计数器
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                no_improve_epochs = 0
                # 保存最佳模型
                torch.save(model.state_dict(), os.path.join(params_dir, 'best_params.pt'))
                torch.save(optimizer.state_dict(), os.path.join(optimizer_dir, 'best_optim.pt'))
            else:
                no_improve_epochs += 1
                
            valloss_file = os.path.join(ckpt_dir, 'valloss.txt') if start_epoch == 0 \
                else os.path.join(ckpt_dir, 'valloss_from_ep{:03d}.txt'.format(start_epoch))

            # 记录验证结果
            if os.path.exists(valloss_file):
                with open(valloss_file, 'a') as f:
                    f.write("ep{:03d} | loss: {:.3f} | valloss: {:.3f} (±{:.3f}) | total_acc: {:.3f} | "
                            "chord_acc: {:.3f} | "
                            "others_acc: {:.3f}\n".format(
                            ep + 1, recons_loss, current_val_loss, np.std(val_recons_losses),
                            np.mean(total_acc_rec), np.mean(chord_acc_rec), np.mean(others_acc_rec)
                    ))
            else:
                with open(valloss_file, 'w') as f:
                    f.write("ep{:03d} | loss: {:.3f} | valloss: {:.3f} (±{:.3f}) | total_acc: {:.3f} | "
                            "chord_acc: {:.3f} | "
                            "others_acc: {:.3f}\n".format(
                            ep + 1, recons_loss, current_val_loss, np.std(val_recons_losses),
                            np.mean(total_acc_rec), np.mean(chord_acc_rec), np.mean(others_acc_rec)
                    ))

            # 检查早停条件
            if no_improve_epochs >= patience:
                print(f'Early stopping triggered after {ep + 1} epochs! No improvement for {no_improve_epochs} epochs.')
                break

        # 打印当前epoch的训练结果
        print('[epoch {:03d}] training completed\n  -- loss = {:.4f}\n  -- time elapsed = {:.2f} secs.'.format(
            ep + 1,
            recons_loss,
            ep_time,
        ))
        # 记录日志
        log_data = {
            'ep': ep + 1,
            'steps': train_steps,
            'ce_loss': recons_loss,
            'time': ep_time
        }
        log_epoch(
            os.path.join(ckpt_dir, log_file), log_data, is_init=not os.path.exists(os.path.join(ckpt_dir, log_file))
        )
