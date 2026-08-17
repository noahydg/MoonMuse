import os
import sys
import time
import yaml
import shutil
import argparse
import random
import numpy as np
from itertools import chain
from collections import defaultdict
import torch

from dataloader import REMISkylineToMidiTransformerDataset, pickle_load
from convert2midi import event_to_midi
from convert_key import degree2pitch, roman2majorDegree, roman2minorDegree

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from generation_budget import (
    RejectionBudget,
    SegmentSamplingBudget,
    monotonic_beat_inadmissibles,
    strip_terminal_eos,
)

sys.path.append('./model')

max_bars = 128
max_dec_inp_len = 2048

emotion_events = ['Emotion_Q1', 'Emotion_Q2', 'Emotion_Q3', 'Emotion_Q4']
samp_per_piece = 1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_inference_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint = {
        key: value for key, value in checkpoint.items()
        if 'feature_map.omega' not in key
    }
    model_state = model.state_dict()
    unexpected = sorted(set(checkpoint) - set(model_state))
    mismatched = sorted(
        key for key in checkpoint
        if key in model_state and checkpoint[key].shape != model_state[key].shape
    )
    missing = sorted(
        key for key in model_state
        if key not in checkpoint and 'feature_map.omega' not in key
    )
    if unexpected or mismatched or missing:
        raise RuntimeError(
            'checkpoint is incompatible: missing={}, unexpected={}, mismatched={}'.format(
                missing[:10], unexpected[:10], mismatched[:10]
            )
        )
    model_state.update(checkpoint)
    model.load_state_dict(model_state, strict=True)
    print('[checkpoint] compatible parameters:', len(checkpoint))

major_map = [0, 4, 7]
minor_map = [0, 3, 7]
diminished_map = [0, 3, 6]
augmented_map = [0, 4, 8]
dominant_map = [0, 4, 7, 10]
major_seventh_map = [0, 4, 7, 11]
minor_seventh_map = [0, 3, 7, 10]
diminished_seventh_map = [0, 3, 6, 9]
half_diminished_seventh_map = [0, 3, 6, 10]
sus_2_map = [0, 2, 7]
sus_4_map = [0, 5, 7]

chord_maps = {
    'M': major_map,
    'm': minor_map,
    'o': diminished_map,
    '+': augmented_map,
    '7': dominant_map,
    'M7': major_seventh_map,
    'm7': minor_seventh_map,
    'o7': diminished_seventh_map,
    '/o7': half_diminished_seventh_map,
    'sus2': sus_2_map,
    'sus4': sus_4_map
}
chord_maps = {k: np.array(v) for k, v in chord_maps.items()}

DEFAULT_SCALE = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
MAJOR_KEY = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
MINOR_KEY = np.array(['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'])


###############################################
# sampling utilities
###############################################
# 速度容忍度
def construct_inadmissible_set(tempo_val, event2idx, tolerance=20):
    inadmissibles = []

    for k, i in event2idx.items():
        if 'Tempo' in k and 'Conti' not in k and abs(int(k.split('_')[-1]) - tempo_val) > tolerance:
            inadmissibles.append(i)

    print(inadmissibles)

    return np.array(inadmissibles)


def temperature(logits, temperature, inadmissibles=12):
    if inadmissibles is not None:
        logits[inadmissibles] -= np.inf

    try:
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        assert np.count_nonzero(np.isnan(probs)) == 0
    except:
        print('overflow detected, use 128-bit')
        # logits = logits.astype(np.float128)
        logits = logits.astype(np.float64)
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        probs = probs.astype(float)
    return probs


def nucleus(probs, p):
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    sorted_index = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_index]
    cutoff = np.searchsorted(np.cumsum(sorted_probs), p, side='left') + 1
    candi_index = sorted_index[:max(1, min(cutoff, len(sorted_index)))]
    candi_probs = probs[candi_index]
    candi_probs /= candi_probs.sum()
    return np.random.choice(candi_index, size=1, p=candi_probs)[0]


##############################################
# data manipulation utilities
##############################################
def merge_tracks(melody_track, chord_track):
    events = melody_track[1:3]

    melody_beat = defaultdict(list)
    if len(melody_track) > 3:
        note_seq = []
        beat = melody_track[3]
        melody_track = melody_track[4:]
        for p in range(len(melody_track)):
            if 'Beat' in melody_track[p]:
                melody_beat[beat] = note_seq
                note_seq = []
                beat = melody_track[p]
            else:
                note_seq.append(melody_track[p])
        melody_beat[beat] = note_seq

    chord_beat = defaultdict(list)
    if len(chord_track) > 2:
        chord_seq = []
        beat = chord_track[2]
        chord_track = chord_track[3:]
        for p in range(len(chord_track)):
            if 'Beat' in chord_track[p]:
                chord_beat[beat] = chord_seq
                chord_seq = []
                beat = chord_track[p]
            else:
                chord_seq.append(chord_track[p])
        chord_beat[beat] = chord_seq

    for b in range(16):
        beat = 'Beat_{}'.format(b)
        if beat in chord_beat or beat in melody_beat:
            events.append(beat)
            if beat in chord_beat:
                events.extend(chord_beat[beat])
            if beat in melody_beat:
                events.extend(melody_beat[beat])

    return events


def read_generated_events(events_file, event2idx):
    events = open(events_file).read().splitlines()
    # print('[test 1] events:', events)
    if 'Key' in events[0]:
        key = events[0]
    else:
        key = 'Key_C'

    bar_pos = np.where(np.array(events) == 'Bar_None')[0].tolist()
    bar_pos.append(len(events))

    chord_bars = []
    for b in range(len(bar_pos)-1):
        chord_bars.append(events[bar_pos[b]: bar_pos[b+1]])

    # print('[test 2] chord_bars:', chord_bars)
    for bar in range(len(chord_bars)):
        chord_bars[bar] = [event2idx[e] for e in chord_bars[bar]]

    return key, chord_bars


def word2event(word_seq, idx2event):
    return [idx2event[w] for w in word_seq]


def extract_midi_events_from_generation(key, events, relative_melody=False):
    if relative_melody:
        new_events = []
        keyname = key.split('_')[1]
        octave = None
        root = None
        quality = None
        for evs in events:
            if 'Note_Octave' in evs:
                octave = int(evs.split('_')[2])
            elif 'Note_Degree' in evs:
                if octave is None:
                    continue
                roman = evs.split('_')[2]
                pitch = degree2pitch(keyname, octave, roman)
                pitch = max(21, pitch)
                pitch = min(108, pitch)
                if pitch < 21 or pitch > 108:
                    raise ValueError('Pitch value must be in (21, 108), but gets {}'.format(pitch))
                new_events.append('Note_Pitch_{}'.format(pitch))
            elif 'Chord_Root' in evs:
                root = evs.split('_')[-1]
            elif 'Chord_Quality' in evs:
                quality = evs.split('_')[-1]
                if root is not None:
                    new_events.append('Chord_{}_{}'.format(root, quality))
                root = None
                quality = None
            # elif 'Chord_' in evs:
            #     if 'None' in evs or 'Conti' in evs:
            #         new_events.append(evs)
            #     else:
            #         root, quality = evs.split('_')[1], evs.split('_')[2]
            #         if keyname in MAJOR_KEY:
            #             root = roman2majorDegree[root]
            #         else:
            #             root = roman2minorDegree[root]
            #         new_events.append('Chord_{}_{}'.format(root, quality))
            else:
                new_events.append(evs)
        events = new_events

    chord_starts = np.where(np.array(events) == 'Track_Chord')[0].tolist()
    melody_starts = np.where(np.array(events) == 'Track_Melody')[0].tolist()

    midi_bars = []
    for st, ed in zip(melody_starts, chord_starts[1:] + [len(events)]):
        bar_midi_events = events[st + 1: ed]
        midi_bars.append(bar_midi_events)

    return midi_bars


def get_position_idx(event):
    return int(event.split('_')[-1])


def event_to_txt(events, output_event_path):
    # 确保目录存在
    os.makedirs(os.path.dirname(output_event_path), exist_ok=True)
    
    # 写入文件
    with open(output_event_path, 'w') as f:
        print(*events, sep='\n', file=f)


# def midi_to_wav(midi_path, output_path):
#     sound_font_path = 'SalamanderGrandPiano-SF2-V3+20200602/SalamanderGrandPiano-V3+20200602.sf2'
#     fs = FluidSynth(sound_font_path)
#     fs.midi_to_audio(midi_path, output_path)


################################################
# main generation function
################################################
def generate_conditional(model, event2idx, idx2event, chord_events, primer,
                         max_events=10000, skip_check=False, max_bars=None,
                         temp=1.2, top_p=0.9, inadmissibles=None,
                         model_type="performer", max_rejections=None,
                         max_segment_attempts=None):
    # 初始化生成的序列，包含primer、LeadSheet轨道标识、LeadSheet事件和Full轨道标识
    generated = primer + [event2idx['Track_Chord']] + chord_events[0] + [event2idx['Track_Melody']]
    # print('[test 0] primer:', primer, 'Track_LeadSheet:', event2idx['Track_LeadSheet'], 'chord_events[0]:', chord_events[0], 'Track_Full:', event2idx['Track_Full'])
    # print('[test 0] generated:', generated)
    # 把generated转化为事件数组
    # generated_events = word2event(generated, idx2event)
    # print('[test 0] generated:', generated_events)
    # 初始化分段输入，用于指示模型的分段信息
    seg_inp = [0 for _ in range(len(generated))]
    seg_inp[-1] = 1  # 标记最后一个事件为新的段

    # 计算目标生成的小节数
    target_bars, generated_bars = len(chord_events), 0
    if max_bars is not None:
        target_bars = min(max_bars, target_bars)  # 如果指定了最大小节数，取最小值

    steps = 0  # 记录生成的步数
    time_st = time.time()  # 记录开始时间
    cur_pos = 0  # 当前事件的位置
    failed_cnt = 0  # 记录生成失败次数
    rejection_budget = RejectionBudget(max_rejections)
    sampling_budget = SegmentSamplingBudget(max_segment_attempts)

    # 开始生成事件，直到生成的小节数达到目标
    while generated_bars < target_bars:
        if sampling_budget.exhausted:
            print('[FATAL] segment sampling attempt limit reached: {}'.format(
                sampling_budget.attempts
            ))
            return generated
        sampling_budget.record()
        assert len(generated) == len(seg_inp)  # 确保生成序列和分段输入长度一致

        # 如果生成的序列长度小于最大解码输入长度，直接使用整个序列
        if len(generated) < max_dec_inp_len:
            dec_input = torch.tensor([generated]).long().to(next(model.parameters()).device)
            dec_seg_inp = torch.tensor([seg_inp]).long().to(next(model.parameters()).device)
        else:
            # 否则，只使用最后max_dec_inp_len个事件
            dec_input = torch.tensor([generated[-max_dec_inp_len:]]).long().to(next(model.parameters()).device)
            dec_seg_inp = torch.tensor([seg_inp[-max_dec_inp_len:]]).long().to(next(model.parameters()).device)

        # 根据模型类型进行前向传播，获取logits
        if model_type == "performer":
            logits = model(
                dec_input,
                seg_inp=dec_seg_inp,
                keep_last_only=True,
                attn_kwargs={'omit_feature_map_draw': steps > 0}
            )
        else:
            logits = model(
                dec_input,
                seg_inp=dec_seg_inp,
                keep_last_only=True,
            )

        # 将logits转换为概率分布，并进行采样
        logits = (logits[0]).cpu().detach().numpy()
        step_inadmissibles = inadmissibles
        if not skip_check:
            step_inadmissibles = monotonic_beat_inadmissibles(
                event2idx, cur_pos, base=inadmissibles
            )
        probs = temperature(logits, temp, inadmissibles=step_inadmissibles)  # 应用温度调节
        word = nucleus(probs, top_p)  # 使用nucleus采样
        word_event = idx2event[word]  # 获取采样到的事件
        # print('[test] (in while loop) word_event:', word_event)

        # 如果不跳过检查，检查事件的位置是否合理
        if not skip_check:
            if 'Beat' in word_event:
                # print('[info] got beat event', word_event)
                event_pos = get_position_idx(word_event)  # 获取事件的位置
                # print('[info]' 'current position:', cur_pos, 'event position:', event_pos)
                if not event_pos >= cur_pos:  # 如果位置没有增加，记录失败次数
                    failed_cnt += 1
                    rejection_limit_reached = rejection_budget.record()
                    print('[info] position not increasing, failed cnt:', failed_cnt, end='\r')
                    if rejection_limit_reached:
                        print('[FATAL] cumulative rejection limit reached: {}'.format(
                            rejection_budget.rejections
                        ))
                        return generated
                    if failed_cnt >= 256:  # 如果失败次数过多，退出生成
                        print('[FATAL] model stuck, exiting with generated events ...')
                        return generated
                    continue
                else:
                    cur_pos = event_pos  # 更新当前位置
                    failed_cnt = 0  # 重置失败次数

        # 如果生成的是LeadSheet轨道标识，表示完成了一个小节的生成
        if word_event == 'Track_Chord':
            steps += 1
            generated.append(word)  # 将事件添加到生成序列
            seg_inp.append(0)  # 更新分段输入
            generated_bars += 1  # 增加生成的小节数
            sampling_budget.reset()
            print('[info] generated {} bars, #events = {}'.format(generated_bars, len(generated)))
            # print('[test 1] generated:', generated)

            # 如果还有小节需要生成，继续添加LeadSheet事件和Full轨道标识
            if generated_bars < target_bars:
                generated.extend(chord_events[generated_bars])
                seg_inp.extend([0 for _ in range(len(chord_events[generated_bars]))])

                generated.append(event2idx['Track_Melody'])
                seg_inp.append(1)
                cur_pos = 0  # 重置当前位置
            continue

        # 如果生成的是PAD或EOS事件，跳过或结束生成
        if word_event == 'PAD_None' or (word_event == 'EOS_None' and generated_bars < target_bars - 1):
            continue
        elif word_event == 'EOS_None' and generated_bars == target_bars - 1:
            print('[info] gotten eos')
            generated.append(word)  # 添加EOS事件
            break

        # 将生成的事件添加到序列中
        generated.append(word)
        # print('[test]' 'generated:', generated)
        seg_inp.append(1)
        steps += 1

        # 如果生成的序列长度超过最大事件数，停止生成
        if len(generated) > max_events:
            print('[info] max events reached')
            break

    # 打印生成结果和时间统计
    print('-- generated events:', len(generated))
    print('-- time elapsed  : {:.2f} secs'.format(time.time() - time_st))
    print('-- time per event: {:.2f} secs'.format((time.time() - time_st) / len(generated)))
    return strip_terminal_eos(generated, idx2event)


if __name__ == '__main__':
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
    parser.add_argument('-i', '--inference_params',
                        help='inference parameters')
    parser.add_argument('-id', '--input_dir',
                        default='generation/stage1_chord',
                        help='conditional input directory')
    parser.add_argument('-od', '--output_dir',
                        default='generation/stage2_melody',
                        help='output directory')
    parser.add_argument('-p', '--play_midi',
                        default=False,
                        help='play midi to audio using FluidSynth', action='store_true')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducible sampling')
    parser.add_argument('--max_bars', type=int, default=max_bars,
                        help='maximum bars to generate per conditional input')
    parser.add_argument('--max_events', type=int, default=10000,
                        help='maximum event tokens to generate per conditional input')
    parser.add_argument('--max_rejections', type=int, default=0,
                        help='maximum cumulative rejected samples per conditional input; 0 disables')
    parser.add_argument('--max_segment_attempts', type=int, default=0,
                        help='maximum model sampling attempts per generated bar; 0 disables')
    parser.add_argument('--temperature', type=float, default=None,
                        help='sampling temperature override')
    parser.add_argument('--top-p', '--top_p', dest='top_p', type=float, default=None,
                        help='nucleus-sampling probability override')
    args = parser.parse_args()

    set_seed(args.seed)
    print('[seed]', args.seed)

    train_conf_path = args.configuration  
    train_conf = yaml.load(open(train_conf_path, 'r'), Loader=yaml.FullLoader)
    print(train_conf)

    representation = args.representation
    if representation == 'remi':
        relative_melody = False
    elif representation == 'functional':
        relative_melody = True

    inference_param_path = args.inference_params
    out_dir = args.output_dir
    input_dir = args.input_dir
    play_midi = args.play_midi

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    train_conf_ = train_conf['training']
    gpuid = train_conf_['gpuid']
    torch.cuda.set_device(gpuid)

    val_split = train_conf['data_loader']['val_split']
    dset = REMISkylineToMidiTransformerDataset(
        train_conf['data_loader']['data_path'].format(representation),
        train_conf['data_loader']['vocab_path'].format(representation),
        model_dec_seqlen=train_conf['model']['max_len'],
        pieces=pickle_load(val_split),
        pad_to_same=True,
    )

    model_conf = train_conf['model']
    model_type = args.model_type
    if model_type == "performer":
        from model.music_performer import MusicPerformer

        model = MusicPerformer(
            dset.vocab_size, model_conf['n_layer'], model_conf['n_head'],
            model_conf['d_model'], model_conf['d_ff'], model_conf['d_embed'],
            use_segment_emb=model_conf['use_segemb'], n_segment_types=model_conf['n_segment_types'],
            favor_feature_dims=model_conf['feature_map']['n_dims']
        ).cuda(gpuid)
        temp, top_p = 1.7, 0.99
    elif model_type == "gpt2":
        # print("[debug] using gpt2 model")
        from model.music_gpt2 import ImprovedMusicGPT2

        model = ImprovedMusicGPT2(
            dset.vocab_size, model_conf['n_layer'], model_conf['n_head'],
            model_conf['d_model'], model_conf['d_ff'], model_conf['d_embed'],
            # use_segment_emb=model_conf['use_segemb'], n_segment_types=model_conf['n_segment_types']
        ).cuda(gpuid)
        temp, top_p = 1.0, 0.97
    else:
        raise NotImplementedError("Unsuppported model:", model_type)
    if args.temperature is not None:
        temp = args.temperature
    if args.top_p is not None:
        top_p = args.top_p
    if temp <= 0:
        parser.error('--temperature must be positive')
    if not 0 < top_p <= 1:
        parser.error('--top-p must be in (0, 1]')
    print(f"[info] temp = {temp} | top_p = {top_p}")

    load_inference_checkpoint(model, inference_param_path)

    model.eval()
    print('[info] model loaded')

    shutil.copy(train_conf_path, os.path.join(out_dir, 'config_melody.yaml'))

    if representation == 'functional':
        files = sorted(i for i in os.listdir(input_dir) if i.endswith('.txt'))
    elif representation in ['remi', 'key']:
        files = sorted(i for i in os.listdir(input_dir) if i.endswith('.txt'))

    for file in files:
        # print('[debug] processing', file)
        # out_name = '_'.join(file.split('/')[-1].split('_')[:2])
        out_name = file.split('.')[0]
        # print(out_name)
        # continue
        if 'Positive' in file:
            emotion_candidate = ['Q1', 'Q4']
        elif 'Negative' in file:
            emotion_candidate = ['Q2', 'Q3']
        elif 'Q1' in file:
            emotion_candidate = ['Q1']
        elif 'Q2' in file:
            emotion_candidate = ['Q2']
        elif 'Q3' in file:
            emotion_candidate = ['Q3']
        elif 'Q4' in file:
            emotion_candidate = ['Q4']
        elif 'None' in file:
            emotion_candidate = ['None']
        else:
            raise ValueError('wrong emotion label')

        for e in emotion_candidate:
            if os.path.exists(os.path.join(out_dir ,out_name + '_' + e + '_melody.txt')):
                print('[info] {} exists, skipping ...'.format(os.path.join(out_dir ,out_name + '_' + e + '_melody.txt')))
                continue
            print(e)
            emotion = dset.event2idx['Emotion_{}'.format(e)]
            # tempo = dset.event2idx['Tempo_{}'.format(110)]
            key, chord_events = read_generated_events(os.path.join(input_dir, file), dset.event2idx)
            if representation in ['functional', 'key']:
                print(key)
                # primer = [emotion, dset.event2idx[key], tempo]
                primer = [emotion, dset.event2idx[key]]
            elif representation == 'remi':
                # primer = [emotion, tempo]
                primer = [emotion]

            with torch.no_grad():
                generated = generate_conditional(model, dset.event2idx, dset.idx2event,
                                                 chord_events, primer=primer,
                                                 max_events=args.max_events,
                                                 max_bars=args.max_bars, temp=temp, top_p=top_p,
                                                 inadmissibles=None, model_type=model_type,
                                                 max_rejections=args.max_rejections or None,
                                                 max_segment_attempts=args.max_segment_attempts or None)

            generated = word2event(generated, dset.idx2event)
            # 把generated保存为txt文件
            output_event_path = os.path.join(out_dir ,out_name + '_' + e + '_melody.txt')
            print('Generated events saved to:', output_event_path)
            event_to_txt(generated, output_event_path)
            generated = extract_midi_events_from_generation(key, generated, relative_melody=relative_melody)

            # output_midi_path = os.path.join(out_dir, out_name + '_' + e + '_melody.mid')
            # event_to_midi(
            #     key,
            #     list(chain(*generated[:max_bars])),
            #     mode='full',
            #     output_midi_path=output_midi_path
            # )

            if play_midi:
                # from midi2audio import FluidSynth

                # output_wav_path = os.path.join(out_dir, out_name + '_' + e + '_chord.wav')
                # midi_to_wav(output_midi_path, output_wav_path)
                pass
