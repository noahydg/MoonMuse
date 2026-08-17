import sys
import os
import random
import pickle
import argparse

import yaml
import torch
import shutil
import numpy as np

from model.plain_transformer import PlainTransformer
from convert2midi import event_to_midi, TempoEvent
from utils import pickle_load
from inference_utils import generate_plain_xl
from convert_key import degree2pitch, roman2majorDegree, roman2minorDegree, MAJOR_KEY

# 添加模型和当前目录到系统路径
sys.path.append('./model/')
sys.path.append('./')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_vocab(vocab_file):
    """
    读取词汇表文件，返回事件到索引、索引到事件的映射，以及词汇表大小。
    
    参数:
        vocab_file (str): 词汇表文件路径
        
    返回:
        event2idx (dict): 事件到索引的映射
        idx2event (dict): 索引到事件的映射
        vocab_size (int): 词汇表大小
    """
    event2idx, idx2event = pickle_load(vocab_file)
    orig_vocab_size = len(event2idx)
    pad_token = orig_vocab_size
    event2idx['PAD_None'] = pad_token  # 添加填充标记
    idx2event[pad_token] = 'PAD_None'
    vocab_size = pad_token + 1

    return event2idx, idx2event, vocab_size


def get_leadsheet_prompt(data_dir, piece, prompt_n_bars):
    """
    从数据目录中读取指定乐曲的提示信息。
    
    参数:
        data_dir (str): 数据目录路径
        piece (str): 乐曲名称
        prompt_n_bars (int): 提示的小节数
        
    返回:
        prompt_evs (list): 提示事件列表
        target_bars (int): 目标小节数
    """
    bar_pos, evs = pickle_load(os.path.join(data_dir, piece))

    prompt_evs = [
        '{}_{}'.format(x['name'], x['value']) for x in evs[: bar_pos[prompt_n_bars] + 1]
    ]
    assert len(np.where(np.array(prompt_evs) == 'Bar_None')[0]) == prompt_n_bars + 1
    target_bars = len(bar_pos)

    return prompt_evs, target_bars


def relative2absolute(key, events):
    """
    将相对音高和和弦转换为绝对音高。
    
    参数:
        key (str): 调性
        events (list): 事件列表
        
    返回:
        events (list): 转换后的事件列表
    """
    new_events = []
    key = key.split('_')[1]
    octave = None
    for evs in events:
        if 'Note_Octave' in evs:
            octave = int(evs.split('_')[2])
        elif 'Note_Degree' in evs:
            if octave is None:
                continue
            roman = evs.split('_')[2]
            pitch = degree2pitch(key, octave, roman)
            pitch = max(21, pitch)
            pitch = min(108, pitch)
            if pitch < 21 or pitch > 108:
                raise ValueError('Pitch value must be in (21, 108), but gets {}'.format(pitch))
            new_events.append('Note_Pitch_{}'.format(pitch))
        elif 'Chord_' in evs:
            if 'None' in evs:
                new_events.append(evs)
            else:
                root, quality = evs.split('_')[1], evs.split('_')[2]
                if key in MAJOR_KEY:
                    root = roman2majorDegree[root]
                else:
                    root = roman2minorDegree[root]
                new_events.append('Chord_{}_{}'.format(root, quality))
        else:
            new_events.append(evs)
    events = new_events

    return events


def event_to_txt(events, output_event_path):
    """
    将事件列表写入文本文件。
    
    参数:
        events (list): 事件列表
        output_event_path (str): 输出文件路径
    """
    f = open(output_event_path, 'w')
    print(*events, sep='\n', file=f)


# def midi_to_wav(midi_path, output_path):
#     """
#     将MIDI文件转换为WAV音频文件。
    
#     参数:
#         midi_path (str): MIDI文件路径
#         output_path (str): 输出WAV文件路径
#     """
#     sound_font_path = 'SalamanderGrandPiano-SF2-V3+20200602/SalamanderGrandPiano-V3+20200602.sf2'
#     fs = FluidSynth(sound_font_path)
#     fs.midi_to_audio(midi_path, output_path)


if __name__ == '__main__':
    # 配置命令行参数
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--configuration',
                          choices=['stage1_chord/config/hooktheory_pretrain.yaml',
                                   'stage1_chord/config/emopia_finetune.yaml',
                                   'stage1_chord/config/pop1k7_pretrain.yaml',
                                   ],
                          default='ckpt/chord/emopia_finetune/config.yaml',
                          help='configurations of training', required=True)
    required.add_argument('-r', '--representation',
                          choices=['remi', 'functional'],
                          default='functional',
                          help='representation for symbolic music', required=True)
    required.add_argument('-m', '--mode',
                          choices=['lead_sheet', 'full_song', 'chord', 'melody', 'performance'],
                          default='chord',
                          help='generation mode', required=True)
    parser.add_argument('-i', '--inference_params',
                        default='ckpt/chord/emopia_finetune/params/best_params.pt',
                        help='inference parameters')
    parser.add_argument('-o', '--output_dir',
                        default='generation/stage1_chord',
                        help='output directory')
    parser.add_argument('-p', '--play_midi',
                        default=False,
                        help='play midi to audio using FluidSynth', action='store_true')
    parser.add_argument('-n', '--n_groups',
                        default=20,
                        help='number of groups to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducible sampling')
    parser.add_argument('--max-bars', '--max_bars', dest='max_bars', type=int,
                        default=None,
                        help='maximum bars to generate; defaults to the mode setting')
    parser.add_argument('--max-events', '--max_events', dest='max_events', type=int,
                        default=None,
                        help='maximum generated event tokens; defaults to the mode setting')
    parser.add_argument('--temperature', type=float, default=None,
                        help='sampling temperature override')
    parser.add_argument('--top-p', '--top_p', dest='top_p', type=float, default=None,
                        help='nucleus-sampling probability override')
    args = parser.parse_args()

    # 加载配置文件
    config_path = args.configuration
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    representation = args.representation # remi 或 functional
    mode = args.mode # lead_sheet 或 full_song
    inference_params = args.inference_params # 预训练模型参数路径：best_weight/Functional-two/emopia_lead_sheet_finetune/ep016_loss0.685_params.pt
    out_dir = args.output_dir # 输出目录
    play_midi = args.play_midi # 是否播放MIDI文件
    n_groups = int(args.n_groups) # 生成乐曲组数：默认20
    set_seed(args.seed)
    print('[seed]', args.seed)
    key_determine = 'rule' # 调性确定方式：rule 或 midi_key
    print('representation: {}, key determine: {}'.format(representation, key_determine))

    # 设置生成参数
    max_bars = 128
    if mode == 'lead_sheet':
        temp = 1.2
        top_p = 0.97
        max_dec_len = 512
        emotions = ['Positive', 'Negative']
    elif mode == 'full_song':
        temp = 1.1
        top_p = 0.99
        max_dec_len = 2400
        emotions = ['Q1', 'Q2', 'Q3', 'Q4']
    elif mode == 'chord':
        temp = 1.1
        top_p = 0.97
        max_dec_len = 128
        emotions = ['Positive', 'Negative']
    if args.max_bars is not None:
        max_bars = args.max_bars
    if args.max_events is not None:
        max_dec_len = args.max_events
    if args.temperature is not None:
        temp = args.temperature
    if args.top_p is not None:
        top_p = args.top_p
    if max_bars <= 0 or max_dec_len <= 0:
        parser.error('--max-bars and --max-events must be positive')
    if temp <= 0:
        parser.error('--temperature must be positive')
    if not 0 < top_p <= 1:
        parser.error('--top-p must be in (0, 1]')
    print('[nucleus parameters] t = {}, p = {}'.format(temp, top_p))
    print('[generation limits] bars = {}, events = {}'.format(max_bars, max_dec_len))

    # 设置GPU设备
    torch.cuda.device(config['device'])

    # 是否使用提示信息
    use_prompt = False
    prompt_bars = 0

    # 创建输出目录
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 读取词汇表
    event2idx, idx2event, vocab_size = read_vocab(config['data']['vocab_path'].format(representation))

    # 如果使用提示信息，加载提示乐曲
    if use_prompt:
        prompt_pieces = pickle_load(config['data']['val_split'])
        prompt_pieces = [x for x in prompt_pieces if os.path.exists(
            os.path.join(config['data']['data_dir'].format(representation), x)
        )]
        if len(prompt_pieces) > n_groups:
            prompt_pieces = random.sample(prompt_pieces, n_groups)

        pickle.dump(
            prompt_pieces,
            open(os.path.join(out_dir, 'sampled_pieces.pkl'), 'wb')
        )
        prompts = []
        for p in prompt_pieces:
            prompts.append(
                get_leadsheet_prompt(
                    config['data']['data_dir'], p,
                    prompt_bars
                )
            )

    # 加载模型配置
    mconf = config['model']
    model = PlainTransformer(
        mconf['d_word_embed'],
        vocab_size,
        mconf['decoder']['n_layer'],
        mconf['decoder']['n_head'],
        mconf['decoder']['d_model'],
        mconf['decoder']['d_ff'],
        mconf['decoder']['tgt_len'],
        mconf['decoder']['tgt_len'],
        dec_dropout=mconf['decoder']['dropout'],
        pre_lnorm=mconf['pre_lnorm']
    ).cuda()
    print('[info] # params:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # 加载预训练模型参数
    pretrained_dict = torch.load(inference_params, map_location='cpu')
    model.load_state_dict(pretrained_dict)
    model.eval()

    # 复制配置文件到输出目录
    if mode == 'lead_sheet':
        shutil.copy(config_path, os.path.join(out_dir, 'config_lead.yaml'))
    elif mode == 'full_song':
        shutil.copy(config_path, os.path.join(out_dir, 'config_full.yaml'))
    elif mode == 'chord':
        shutil.copy(config_path, os.path.join(out_dir, 'config_chord.yaml'))
    elif mode =='melody':
        shutil.copy(config_path, os.path.join(out_dir, 'config_melody.yaml'))
    elif mode == 'performance':
        shutil.copy(config_path, os.path.join(out_dir, 'config_performance.yaml'))

    # 生成音乐
    generated_pieces = 0
    total_pieces = n_groups
    gen_times = []

    while generated_pieces < n_groups:
        for emotion in emotions:
            out_name = 'samp_{:02d}_{}'.format(generated_pieces, emotion)
            print(out_name)
            if os.path.exists(os.path.join(out_dir, out_name + '.mid')):
                print('[info] {} exists, skipping ...'.format(out_name))
                continue

            if not use_prompt:
                tempo = 110
                orig_tempos = [TempoEvent(tempo, 0, 0)]
                print('[global tempo]', orig_tempos[0].tempo)
            else:
                target_bars = prompts[generated_pieces][1]
                tempo = 110
                orig_tempos = [TempoEvent(tempo, 0, 0)]

            # generate music events
            if not use_prompt:
                gen_words, t_sec = generate_plain_xl(
                                      model,
                                      event2idx, idx2event,
                                      max_events=max_dec_len, max_bars=max_bars,
                                      primer=['Emotion_{}'.format(emotion)],
                                      temp=temp, top_p=top_p,
                                      representation=representation,
                                      key_determine=key_determine
                                    )
            else:
                gen_words, t_sec = generate_plain_xl(
                                      model,
                                      event2idx, idx2event,
                                      max_events=max_dec_len, max_bars=target_bars,
                                      primer=['Emotion_{}'.format(emotion)] + prompts[generated_pieces][0][1:],
                                      temp=temp, top_p=top_p,
                                      prompt_bars=prompt_bars,
                                      representation=representation,
                                      key_determine=key_determine
                                    )

            gen_words = [idx2event[w] for w in gen_words]

            # 确定调性
            key = None
            for evs in gen_words:
                if 'Key' in evs:
                    key = evs
            if key is None:
                key = 'Key_C'

            # 将相对音高转换为绝对音高
            if mode == 'chord':
                gen_words = gen_words[1:]
            elif representation == 'functional':
                gen_words_roman = gen_words[1:]
                gen_words = relative2absolute(key, gen_words)[1:]
            else:
                gen_words = gen_words[1:]

            if gen_words is None:  # 模型生成失败
                continue

            # 将生成的事件转换为MIDI文件
            if mode == 'lead_sheet':
                event_to_midi(key, gen_words, mode=mode,
                              output_midi_path=os.path.join(out_dir, out_name + '.mid'),
                              play_chords=True, enforce_tempo=True, enforce_tempo_evs=orig_tempos)
            elif mode == 'full_song':
                event_to_midi(key, gen_words, mode=mode,
                              output_midi_path=os.path.join(out_dir, out_name + '.mid'))

            if mode == 'chord':
                event_to_txt(gen_words, output_event_path=os.path.join(out_dir, out_name + '.txt'))
            elif representation == 'functional':
                event_to_txt(gen_words_roman, output_event_path=os.path.join(out_dir, out_name + '_roman.txt'))
            else:
                event_to_txt(gen_words, output_event_path=os.path.join(out_dir, out_name + '.txt'))

            gen_times.append(t_sec)

            # # 如果需要，将MIDI转换为WAV音频
            # if play_midi:
            #     from midi2audio import FluidSynth

            #     output_midi_path = os.path.join(out_dir, out_name + '.mid')
            #     output_wav_path = os.path.join(out_dir, out_name + '.wav')
            #     midi_to_wav(output_midi_path, output_wav_path)

        generated_pieces += 1

    # 输出生成结果
    print('[info] finished generating {} pieces, avg. time: {:.2f} +/- {:.2f} secs.'.format(
        generated_pieces, np.mean(gen_times), np.std(gen_times)
    ))
