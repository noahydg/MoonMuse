# 导入所需库
import os
import csv
import gzip
import json
import random
import numpy as np
import collections
from tqdm import tqdm

# 数据文件路径
emopia_data_home = 'H:/PyHome/music/workdir/dataset/EMOPIA+/'  # EMOPIA+ 数据集的路径
hooktheory_data_home = 'H:/PyHome/music/workdir/dataset/HookTheory/'  # HookTheory 数据集的路径

# 音乐调式（大调、小调）
MAJOR_KEY = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])  # 大调
MINOR_KEY = np.array(['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'])  # 小调

# 音符对应的音高索引（大调）
IDX_TO_KEY = {
    9: 'A',
    10: 'A#',
    11: 'B',
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#'
}
KEY_TO_IDX = {v: k for k, v in IDX_TO_KEY.items()}  # 构建反向映射

# 大调和弦罗马数字的映射
majorDegree2roman = {
    0: 'I',
    1: 'I#',
    2: 'II',
    3: 'II#',
    4: 'III',
    5: 'IV',
    6: 'IV#',
    7: 'V',
    8: 'V#',
    9: 'VI',
    10: 'VI#',
    11: 'VII',
}
roman2majorDegree = {v: k for k, v in majorDegree2roman.items()} # 构建反向映射

minorDegree2roman = {
    0: 'I',
    1: 'I#',
    2: 'II',
    3: 'III',
    4: random.choice(['III', 'IV']),
    5: 'IV',
    6: 'IV#',
    7: 'V',
    8: 'VI',
    9: 'VI#',
    10: 'VII',
    11: random.choice(['VII', 'I'])
}
roman2minorDegree = {
    'I': 0,
    'I#': 1,
    'II': 2,
    'II#': random.choice([2, 3]),
    'III': 3,
    'IV': 5,
    'IV#': 6,
    'V': 7,
    'V#': random.choice([7, 8]),
    'VI': 8,
    'VI#': 9,
    'VII': 10
}

# 读取 EMOPIA+ 数据集中的调式信息
def find_key_emopia():
    print('load keyname for emopia clips ...')  # 打印加载信息
    header, content = csv_read(os.path.join(emopia_data_home, 'key_mode_tempo.csv'))  # 读取csv文件
    clip2keyname = collections.defaultdict(str)  # 默认值为空字符串的字典，用于存储片段与调式的映射
    clip2keymode = collections.defaultdict(str)  # 默认值为空字符串的字典，用于存储片段与调式类型（大调或小调）的映射
    for c in content:
        name = c[1]  # 获取片段名
        keyname = c[2]  # 获取调名
        keymode = 0 if keyname in MAJOR_KEY else 1  # 如果是大调，标记为0；否则标记为小调（1）
        clip2keyname[name] = keyname  # 存储片段名与调名的映射
        clip2keymode[name] = keymode  # 存储片段名与调式类型的映射
    return clip2keyname, clip2keymode  # 返回调名和调式类型的映射

# 读取 HookTheory 数据集中的调式信息
def find_key_hooktheory():
    print('load keyname for HookTheory clips ...')  # 打印加载信息
    with gzip.open(os.path.join(hooktheory_data_home, 'Hooktheory.json.gz'), 'r') as f:  # 打开并读取gzip压缩的JSON文件
        dataset = json.load(f)  # 解析JSON文件

    clip2keyname = dict()  # 存储片段与调名的映射
    clip2keymode = dict()  # 存储片段与调式类型的映射
    for k, v in tqdm(dataset.items()):  # 遍历数据集中的每一项
        clip_name = k  # 片段名
        annotations = v['annotations']  # 获取片段的注释信息
        key = IDX_TO_KEY[annotations['keys'][0]['tonic_pitch_class']]  # 获取音调名称
        mode = list2str(annotations['keys'][0]['scale_degree_intervals'])  # 获取调式结构（例如：221222）

        # 判断调式结构，确定是大调还是小调
        if mode == '221222':
            clip2keyname[clip_name] = key.upper()  # 大调
            clip2keymode[clip_name] = 0  # 标记为大调
        elif mode == '212212':
            clip2keyname[clip_name] = key.lower()  # 小调
            clip2keymode[clip_name] = 1  # 标记为小调
        else:
            continue  # 如果不是大调或小调，跳过

    return clip2keyname, clip2keymode  # 返回调名和调式类型的映射

# 将音高转换为音符的度数和罗马数字表示
def pitch2degree(key, pitch):
    degree = pitch % 12  # 计算音符在十二度音阶中的位置

    # 如果是大调
    if key in MAJOR_KEY:
        tonic = KEY_TO_IDX[key]  # 获取主音的索引
        degree = (degree + 12 - tonic) % 12  # 计算度数
        octave = (pitch - degree) // 12  # 计算八度
        roman = majorDegree2roman[degree]  # 获取罗马数字表示
    # 如果是小调
    elif key in MINOR_KEY:
        tonic = KEY_TO_IDX[key.upper()]  # 获取主音的索引（大调的索引）
        degree = (degree + 12 - tonic) % 12  # 计算度数
        octave = (pitch - degree) // 12  # 计算八度
        roman = minorDegree2roman[degree]  # 获取罗马数字表示
    else:
        raise NameError('Wrong key name {}.'.format(key))  # 如果调式名称错误，抛出异常

    return octave, roman  # 返回八度和罗马数字表示

# 将音符的度数和罗马数字表示转换为音高
def degree2pitch(key, octave, roman):
    # 如果是大调
    if key in MAJOR_KEY:
        tonic = KEY_TO_IDX[key]  # 获取主音的索引
        pitch = octave * 12 + tonic + roman2majorDegree[roman]  # 计算音高
    # 如果是小调
    elif key in MINOR_KEY:
        tonic = KEY_TO_IDX[key.upper()]  # 获取主音的索引（大调的索引）
        pitch = octave * 12 + tonic + roman2minorDegree[roman]  # 计算音高
    else:
        raise NameError('Wrong key name {}.'.format(key))  # 如果调式名称错误，抛出异常

    return pitch  # 返回计算得到的音高

# 将绝对音符表示转换为相对音符表示
def absolute2relative(events, enforce_key=False, enforce_key_evs=None):
    if enforce_key:
        key = enforce_key_evs['value']  # 如果强制指定调式，则使用强制指定的调式
    else:
        for evs in events:
            if evs['name'] == 'Key':  # 查找调式信息
                key = evs['value']
                break

    new_events = []  # 存储转换后的事件列表
    for evs in events:
        if evs['name'] == 'Key':
            new_events.append({'name': 'Key', 'value': key})  # 调式事件不变
        elif evs['name'] == 'Note_Pitch':
            pitch = evs['value']  # 获取音高
            octave, roman = pitch2degree(key, pitch)  # 转换为度数和罗马数字
            new_events.append({'name': 'Note_Octave', 'value': octave})  # 添加八度事件
            new_events.append({'name': 'Note_Degree', 'value': roman})  # 添加度数事件
        else:
            new_events.append(evs)  # 其他事件保持不变

    return new_events  # 返回转换后的事件列表

# 将相对音符表示转换为绝对音符表示
def relative2absolute(events, enforce_key=False, enforce_key_evs=None):
    if enforce_key:
        key = enforce_key_evs['value']  # 如果强制指定调式，则使用强制指定的调式
    else:
        for evs in events:
            if evs['name'] == 'Key':  # 查找调式信息
                key = evs['value']
                break

    new_events = []  # 存储转换后的事件列表
    for evs in events:
        if evs['name'] == 'Key':
            new_events.append({'name': 'Key', 'value': key})  # 调式事件不变
        elif evs['name'] == 'Note_Octave':
            octave = evs['value']  # 获取八度
        elif evs['name'] == 'Note_Degree':
            roman = evs['value']  # 获取度数（罗马数字）
            pitch = degree2pitch(key, octave, roman)  # 转换为音高
            pitch = max(21, pitch)  # 限制音高范围
            pitch = min(108, pitch)
            if pitch < 21 or pitch > 108:
                raise ValueError('Pitch value must be in (21, 108), but gets {}'.format(pitch))  # 校验音高范围
            new_events.append({'name': 'Note_Pitch', 'value': pitch})  # 添加音高事件
        else:
            new_events.append(evs)  # 其他事件保持不变

    return new_events  # 返回转换后的事件列表

# 切换调式（大调和小调）
def switch_key(key):
    if '_' in key:
        keyname = key.split('_')[1]  # 获取调式名称（去掉前缀）
        if keyname in MAJOR_KEY:
            return 'Key_' + keyname.lower()  # 大调转小调
        if keyname in MINOR_KEY:
            return 'Key_' + keyname.upper()  # 小调转大调
    if key in MAJOR_KEY:
        return key.lower()  # 大调转小调
    if key in MINOR_KEY:
        return key.upper()  # 小调转大调

# 根据片段的名称和事件切换旋律（调式）
def switch_melody(filename, events, clip2keymode):
    keymode = int(clip2keymode[filename])  # 获取该片段的调式类型
    # 如果调式不需要切换，则返回原始事件
    if (filename[:2] in ['Q1', 'Q4'] and keymode == 1) or (filename[:2] in ['Q2', 'Q3'] and keymode == 0):
        return events
    # 否则，切换调式
    else:
        keyname = 'C' if keymode == 0 else 'c'  # 默认调式
        key_event = {'name': 'Key', 'value': keyname}
        new_events = absolute2relative(events, enforce_key=True, enforce_key_evs=key_event)  # 转换为相对表示

        new_key_event = {'name': 'Key', 'value': switch_key(keyname)}  # 切换调式
        new_events = relative2absolute(new_events, enforce_key=True, enforce_key_evs=new_key_event)  # 转换回绝对表示
        return new_events  # 返回转换后的事件列表

# 读取csv文件的函数
def csv_read(path):
    content = list()  # 存储读取的数据
    with open(path, 'r') as f:
        reader = csv.reader(f)  # 使用csv模块读取文件
        for row in reader:
            content.append(row)  # 将每一行数据加入列表
    f.close()  # 关闭文件
    header = content[0]  # 第一行作为表头
    content = content[1:]  # 后续行作为内容
    return header, content  # 返回表头和内容

# 将列表转换为字符串
def list2str(a_list):
    return ''.join([str(i) for i in a_list])  # 将列表元素拼接为字符串
