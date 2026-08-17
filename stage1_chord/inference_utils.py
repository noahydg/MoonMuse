import os
import sys
import time

import torch
import numpy as np

from utils import tensor_to_numpy
from convert_key import MINOR_KEY, MAJOR_KEY

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from generation_budget import strip_terminal_eos


########################################
# 采样工具函数
########################################

def temperature(logits, temperature):
    """
    使用温度参数对logits进行缩放，并计算概率分布。
    
    参数:
    - logits: 模型的原始输出（未归一化的对数概率）。
    - temperature: 温度参数，控制概率分布的平滑程度。温度越高，分布越平滑；温度越低，分布越尖锐。
    
    返回:
    - probs: 经过温度缩放后的概率分布。
    """
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= np.max(scaled)
    probs = np.exp(scaled)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError('invalid probability distribution')
    return probs / total


def nucleus(probs, p):
    """
    使用nucleus sampling（也称为top-p sampling）从概率分布中采样。
    
    参数:
    - probs: 概率分布。
    - p: 累积概率阈值，只从累积概率超过p的最小集合中采样。
    
    返回:
    - word: 采样得到的索引。
    """
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    sorted_index = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_index]
    cutoff = np.searchsorted(np.cumsum(sorted_probs), p, side='left') + 1
    candi_index = sorted_index[:max(1, min(cutoff, len(sorted_index)))]
    candi_probs = probs[candi_index]
    candi_probs /= candi_probs.sum()
    return np.random.choice(candi_index, size=1, p=candi_probs)[0]


def get_position_idx(event):
    """
    从事件名称中提取位置索引。
    
    参数:
    - event: 事件名称，格式为"event_position"。
    
    返回:
    - position_idx: 事件名称中的位置索引。
    """
    return int(event.split('_')[-1])

########################################
# main inference driver
########################################
def generate_plain_xl(model, event2idx, idx2event, max_bars=160,
                      max_events=2048, primer=None, temp=1.2, top_p=0.9,
                      prompt_bars=None, representation='functional', key_determine=None):
    if primer is None: # 无先前生成的音符
        generated = [event2idx['Bar_None']]
        # print('[test 1] generated:', generated)
        target_bars, generated_bars = max_bars, 0
    else: # 有先前生成的音符
        generated = [event2idx[e] for e in primer]
        # print('[test 2] generated:', generated)
        target_bars, generated_bars = max_bars, prompt_bars if prompt_bars is not None else 0

    device = next(model.parameters()).device
    steps = 0 # 计步数
    time_st = time.time() # 计时
    cur_pos = 0 # 当前音符位置
    failed_cnt = 0 # 连续失败次数
    mems = tuple() # 记忆单元
    while generated_bars < target_bars: # 循环生成音符
        if steps == 0: # 首次生成
            dec_input = torch.LongTensor([generated]).to(device) 
            dec_input = dec_input.permute(1, 0) if len(generated) > 1 else dec_input 
            # print ('[shape]', dec_input.size(), dec_seg_emb.size())
        else:
            dec_input = torch.LongTensor([[generated[-1]]]).to(device)
        # print (dec_input.size(), dec_seg_emb.size())
        # print('[test 4] dec_input:', dec_input)

        # sampling
        logits, mems = model.generate(dec_input, mems) # 生成下一个token的logits
        # print('[test 5] logits:', logits)
        # print('[test 5] mems[0].size():', mems[0].size())
        logits = tensor_to_numpy(logits)

        if representation in ['functional', 'key'] and len(generated) == 1:
            if key_determine == 'rule':
                emotion_label = idx2event[generated[0]].split('_')[1]
                allowed_keys = [
                    idx for event, idx in event2idx.items()
                    if event.startswith('Key_')
                    and match_emotion_key(emotion_label, event.split('_', 1)[1])
                ]
                if not allowed_keys:
                    raise ValueError(
                        '[info] no key tokens match emotion {}'.format(emotion_label)
                    )
                constrained_logits = np.full_like(logits, -np.inf, dtype=np.float64)
                constrained_logits[allowed_keys] = logits[allowed_keys]
                probs = temperature(constrained_logits, temperature=1.1)
            else:
                probs = temperature(logits, temperature=1.1)
            word = nucleus(probs, p=0.97)
            word_event = idx2event[word] 
            # print('[test] word_event:', word_event)
        else:
            probs = temperature(logits, temperature=temp)
            word = nucleus(probs, p=top_p)
            # print('[test] word(else):', word)
            word_event = idx2event[word]
            # print('[test] word_event(else):', word_event)
            # print(temp, top_p)

        if 'Key' in word_event:
            print('[info] generated {}, #events = {}'.format(word_event, len(generated)))

        if 'Beat' in word_event:
            event_pos = get_position_idx(word_event)
            if not event_pos >= cur_pos:
                failed_cnt += 1
                print('[info] position not increasing, failed cnt:', failed_cnt, end='\r')
                if failed_cnt >= 256:
                    print('[FATAL] model stuck, exiting ...')
                    return None, time.time() - time_st
                continue
            else:
                cur_pos = event_pos
                failed_cnt = 0

        if 'Bar' in word_event:
            generated_bars += 1
            cur_pos = 0
            print('[info] generated {} bars, #events = {}'.format(generated_bars, len(generated)))
        if word_event == 'PAD_None':
            continue

        generated.append(word)
        # print('[test 3] generated:', generated)
        # print ([idx2event[e] for e in generated])
        steps += 1

        if len(generated) > max_events:
            print('[info] max events reached')
            break
        if word_event == 'EOS_None':
            print('[info] gotten eos')
            break

    print('-- generated events:', len(generated))
    print('-- time elapsed: {:.2f} secs'.format(time.time() - time_st))

    return strip_terminal_eos(generated, idx2event), time.time() - time_st


def match_emotion_key(emotion, key):
    if emotion in ['Q1', 'Q4', 'Positive'] and key in MAJOR_KEY:
        return True
    if emotion in ['Q2', 'Q3', 'Negative'] and key in MINOR_KEY:
        return True
    return False
