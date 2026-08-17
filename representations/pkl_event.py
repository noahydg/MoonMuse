import pickle
from tqdm import tqdm
import os

add_track = False  

def find_idx(data, mode='chord'):
    # 初始化一个空列表来存储索引
    bar_idx = []

    if mode == 'chord' or mode == 'melody' or mode == 'performance':
        # 遍历数据列表
        for index, item in enumerate(data):
            # 检查 name 是否为 'Bar'
            if item['name'] == 'Bar':
                # 如果是，将索引添加到列表中
                bar_idx.append(index)
    
    elif mode == 'c2m':
        # 找出所有Track的位置和类型
        track_indices = []
        eos_index = len(data)  # 默认EOS位置为数据末尾+1
        
        for index, item in enumerate(data):
            if item['name'] == 'Track' and item['value'] in ['Chord', 'Melody']:
                track_indices.append((index, item['value']))
            elif item.get('name') == 'EOS':  # 检查EOS标记
                eos_index = index
        
        # 生成两种配对
        chord_melody_pairs = []
        melody_chord_pairs = []
        
        for i in range(len(track_indices)):
            current_idx, current_type = track_indices[i]
            
            if i < len(track_indices)-1:
                next_idx, next_type = track_indices[i+1]
            else:
                # 最后一个元素，使用EOS位置
                next_idx = eos_index
                next_type = 'EOS'
            
            if current_type == 'Chord' and next_type == 'Melody':
                chord_melody_pairs.append((current_idx, next_idx))
            elif current_type == 'Melody' and next_type == 'Chord':
                melody_chord_pairs.append((current_idx, next_idx))
            elif current_type == 'Melody' and next_type == 'EOS':
                # 处理最后一个Melody没有配对的Chord的情况
                melody_chord_pairs.append((current_idx, next_idx))
        
        return chord_melody_pairs, melody_chord_pairs
    
    elif mode == 'm2p':
        # 找出所有Track的位置和类型
        track_indices = []
        eos_index = len(data)  # 默认EOS位置为数据末尾+1
        
        for index, item in enumerate(data):
            if item['name'] == 'Track' and item['value'] in ['Melody', 'Performance']:
                track_indices.append((index, item['value']))
            elif item.get('name') == 'EOS':  # 检查EOS标记
                eos_index = index
        
        # 生成两种配对
        melody_perf_pairs = []
        perf_melody_pairs = []
        
        for i in range(len(track_indices)):
            current_idx, current_type = track_indices[i]
            
            if i < len(track_indices)-1:
                next_idx, next_type = track_indices[i+1]
            else:
                # 最后一个元素，使用EOS位置
                next_idx = eos_index
                next_type = 'EOS'
            
            if current_type == 'Melody' and next_type == 'Performance':
                melody_perf_pairs.append((current_idx, next_idx))
            elif current_type == 'Performance' and next_type == 'Melody':
                perf_melody_pairs.append((current_idx, next_idx))
            elif current_type == 'Performance' and next_type == 'EOS':
                # 处理最后一个Performance没有配对的Melody的情况
                perf_melody_pairs.append((current_idx, next_idx))
        
        return melody_perf_pairs, perf_melody_pairs
    
    # 返回包含所有 'Bar' 索引的列表
    return [bar_idx]

def pkl2leadsheet(file_path, output_path):
    """
    从 .pkl 文件中提取 LeadSheet 信息，仅对 Track 为 'Full' 的层级去掉 Note_Duration、Tempo 和 Note_Velocity，
    并保存到新文件。

    :param file_path: 输入的 .pkl 文件路径
    :param output_path: 输出的 .pkl 文件路径
    """
    # 从 .pkl 文件中加载数据
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # 提取包含 'name' 和 'value' 的字典列表
    name_value_list = data[2]

    # 创建一个新的列表，用于存储符合条件的数据
    new_data = []

    # 将前三行直接加入新数据
    new_data.extend(name_value_list[:3])

    # 从第四行开始遍历
    i = 3
    while i < len(name_value_list):
        item = name_value_list[i]
        
        # 如果 Track 为 'Full'
        if item['name'] == 'Track' and item['value'] == 'Full':
            # 将当前行添加到新数据中
            item['value'] = 'LeadSheet'
            new_data.append(item)
            i += 1
            
            # 进入循环，处理后续行
            while i < len(name_value_list):
                next_item = name_value_list[i]
                
                # 如果事件为 Bar
                if next_item['name'] == 'Bar':
                    new_data.append(next_item)
                    i += 1
                
                # 如果事件为 Beat
                elif next_item['name'] == 'Beat':
                    new_data.append(next_item)
                    i += 1
                    
                    # 捕捉第一个连续的 Note_Octave, Note_Degree, Note_Duration
                    if (i + 2 < len(name_value_list)) and \
                    (name_value_list[i]['name'] == 'Note_Octave') and \
                    (name_value_list[i + 1]['name'] == 'Note_Degree') and \
                    (name_value_list[i + 2]['name'] == 'Note_Duration'):
                        # 将 Note_Octave, Note_Degree, Note_Duration 加入新数据
                        new_data.append(name_value_list[i])
                        new_data.append(name_value_list[i + 1])
                        new_data.append(name_value_list[i + 2])
                        i += 3  # 跳过这三行
                    
                    # 忽略后续的 Note_Octave, Note_Degree, Note_Duration 和 Note_Velocity
                    while i < len(name_value_list):
                        if name_value_list[i]['name'] in ['Bar', 'Beat', 'Chord']:
                            break  # 遇到 Bar、Beat 或 Chord，恢复捕捉
                        i += 1  # 否则跳过当前行
                
                # 如果事件为 Chord
                elif next_item['name'] == 'Chord':
                    new_data.append(next_item)
                    i += 1
                    
                    # 捕捉紧接着的 Note_Octave, Note_Degree, Note_Duration
                    if (i + 2 < len(name_value_list)) and \
                    (name_value_list[i]['name'] == 'Note_Octave') and \
                    (name_value_list[i + 1]['name'] == 'Note_Degree') and \
                    (name_value_list[i + 2]['name'] == 'Note_Duration'):
                        # 将 Note_Octave, Note_Degree, Note_Duration 加入新数据
                        new_data.append(name_value_list[i])
                        new_data.append(name_value_list[i + 1])
                        new_data.append(name_value_list[i + 2])
                        i += 3  # 跳过这三行
                    
                    # 忽略后续的 Note_Octave, Note_Degree, Note_Duration 和 Note_Velocity
                    while i < len(name_value_list):
                        if name_value_list[i]['name'] in ['Bar', 'Beat', 'Chord']:
                            break  # 遇到 Bar、Beat 或 Chord，恢复捕捉
                        i += 1  # 否则跳过当前行
                
                # 忽略 Tempo 和 Note_Velocity
                elif next_item['name'] in ['Tempo', 'Note_Velocity']:
                    i += 1  # 跳过当前行
                
                # 其他情况跳过
                else:
                    i += 1
        else:
            i += 1

    # 将新数据保存到输出文件，确保数据结构为 [None, None, new_data]
    with open(output_path, 'wb') as f:
        pickle.dump([None, None, new_data], f)

def pkl2melody(file_path, output_path, pos=2):
    """
    从 .pkl 文件中提取旋律信息，仅对 Track 为 'Full' 的层级去掉 Note_Duration、Tempo 和 Note_Velocity，
    并保存到新文件。

    :param file_path: 输入的 .pkl 文件路径
    :param output_path: 输出的 .pkl 文件路径
    """
    # 从 .pkl 文件中加载数据
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # 提取包含 'name' 和 'value' 的字典列表
    name_value_list = data[pos]

    # 创建一个新的列表，用于存储符合条件的数据
    new_data = []

    # 将前三行直接加入新数据
    new_data.extend(name_value_list[:3])

    # 从第四行开始遍历
    i = 3
    while i < len(name_value_list):
        item = name_value_list[i]
        
        # 如果 Track 为 'Full'
        if item['name'] == 'Track' and item['value'] == 'Full':
            # 将当前行添加到新数据中
            item['value'] = 'Melody'
            new_data.append(item)
            i += 1
            
            # 进入 Track 为 'Full' 的逻辑
            while i < len(name_value_list):
                next_item = name_value_list[i]
                
                # 如果事件不是 Note_Duration、Tempo 或 Note_Velocity，则加入新数据
                if next_item['name'] not in ['Note_Duration', 'Tempo', 'Note_Velocity']:
                    new_data.append(next_item)
                
                i += 1  # 移动到下一行
        else:
            # 如果 Track 不是 'Full'，则跳过当前行
            i += 1

    # 将新数据保存到输出文件，确保数据结构为 [None, None, new_data]
    with open(output_path, 'wb') as f:
        pickle.dump([None, None, new_data], f)

def pkl2chord(file_path, output_path, pos=2, check_track=True):
    """
    从 .pkl 文件中提取和弦信息，仅对 Track 为 'Full' 的层级去掉 Note_*、Tempo 和 Note_Velocity，
    并保存到新文件。
    或直接去掉 Note_*、Tempo 和 Note_Velocity。

    :param file_path: 输入的 .pkl 文件路径
    :param output_path: 输出的 .pkl 文件路径
    """
    # 从 .pkl 文件中加载数据
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # 提取包含 'name' 和 'value' 的字典列表
    name_value_list = data[pos]

    # 创建一个新的列表，用于存储符合条件的数据
    new_data = []

    item = name_value_list[0]
    if item['name'] == 'Emotion':
        if item['value'] in ['Q1','Q4']:
            item['value'] = 'Positive'
        elif item['value'] in ['Q2','Q3']:
            item['value'] = 'Negative'

    new_data.append(item)

    for i in range(1,3):
        item = name_value_list[i]
        if item['name'] not in ['Tempo', 'Beat', 'Velocity']:
            new_data.append(item)

    # 从第四行开始遍历
    i = 3
    chord_cache_root, chord_cache_quality = None, None
    if check_track:
        while i < len(name_value_list):
            item = name_value_list[i]
            
            # 如果 Track 为 'Full'
            if item['name'] == 'Track' and item['value'] == 'Full':
                # 将当前行添加到新数据中
                if add_track:
                    item['value'] = 'Chord'
                    new_data.append(item)
                i += 1
                
                # 进入 Track 为 'Full' 的逻辑
                while i < len(name_value_list):
                    next_item = name_value_list[i]
            
                    # 如果事件不是 Note_Duration、Tempo 或 Note_Velocity，则加入新数据
                    if next_item['name'] == 'Track' and next_item['value'] == 'LeadSheet':
                        break
                    elif not next_item['name'].startswith('Note_') and next_item['name'] not in ['Tempo', 'Beat']:
                        new_data.append(next_item)
                    
                    i += 1  # 移动到下一行
            else:
                # 如果 Track 不是 'Full'，则跳过当前行
                i += 1
    else:
        while i < len(name_value_list):
            item = name_value_list[i]
            if not item['name'].startswith('Note_') and item['name'] not in ['Tempo', 'Beat', 'Velocity', 'Track']:
                if item['name'] == 'Chord_Root' and item['value'] == chord_cache_root:
                    item['value'] = 'Conti'
                else:
                    chord_cache_root = item['value']
                if item['name'] == 'Chord_Quality' and item['value'] == chord_cache_quality:
                    item['value'] = 'Conti'
                else:
                    chord_cache_quality = item['value']

                new_data.append(item)
            i += 1
    # 将新数据保存到输出文件，
    item['name'] = 'EOS'
    item['value'] = None
    new_data.append(item)
    with open(output_path, 'wb') as f:
        pickle.dump([find_idx(new_data), new_data], f)

def process_folder_to_melody(file_dir, output_dir, pos=2):
    """
    处理文件夹下所有 .pkl 文件为 melody。

    :param file_dir: 输入的文件夹路径
    :param output_dir: 输出的文件夹路径
    """
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历文件夹下所有 .pkl 文件
    for file_name in tqdm(os.listdir(file_dir), desc="处理melody文件中"):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(file_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            pkl2melody(file_path, output_path, pos=pos)

def process_folder_to_chord(file_dir, output_dir, pos=2 ,check_track=True):
    """
    处理文件夹下所有 .pkl 文件为 chord。

    :param file_dir: 输入的文件夹路径
    :param output_dir: 输出的文件夹路径
    """
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历文件夹下所有 .pkl 文件
    for file_name in tqdm(os.listdir(file_dir), desc="处理 chord文件中"):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(file_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            pkl2chord(file_path, output_path, pos=pos, check_track=check_track)

def show_data(data,n):
    count = 0
    for item in data:
        print(item)
        count += 1
        if count > n:
            break

def ensure_beat_values(data):
    """
    检查并修复数据结构，确保每个Bar后面、下一个Bar之前都有Beat值为0,4,8,12的行
    如果缺少则补充相应的行，并按正确顺序插入
    """
    new_data = data.copy()
    i = 0
    n = len(new_data)
    
    while i < n:
        current_item = new_data[i]
        
        if current_item['name'] == 'Bar':
            # 收集直到下一个Bar或结尾的所有Beat信息
            beats = []
            j = i + 1
            next_bar_pos = n
            
            while j < n:
                if new_data[j]['name'] == 'Bar':
                    next_bar_pos = j
                    break
                if new_data[j]['name'] == 'Beat':
                    beats.append((j, new_data[j]['value']))
                j += 1
            
            # 检查缺失的Beat
            required = {0, 4, 8, 12}
            existing = {value for (pos, value) in beats}
            missing = sorted(required - existing)
            
            if missing:
                # 按顺序处理缺失的Beat
                for beat in missing:
                    # 找到正确的插入位置
                    insert_pos = next_bar_pos
                    for k in range(i + 1, next_bar_pos):
                        if new_data[k]['name'] == 'Beat' and new_data[k]['value'] > beat:
                            insert_pos = k
                            break
                    
                    # 插入Beat和Chord行
                    new_data.insert(insert_pos, {'name': 'Beat', 'value': beat})
                    new_data.insert(insert_pos + 1, {'name': 'Chord', 'value': 'Conti_Conti'})
                    n += 2
                    next_bar_pos += 2
                    
                    # 如果插入位置在当前索引之前，需要调整索引
                    if insert_pos <= i:
                        i += 2
        
        i += 1
    
    return new_data

def split_chord_into_root_quality(data):
    """
    将每个Chord行拆分为Chord_Root和Chord_Quality两行
    """
    new_data = []
    
    for item in data:
        if item['name'] == 'Chord':
            # 优化分割逻辑，避免多次split
            chord_value = item['value']
            split_pos = chord_value.find('_')
            
            if split_pos == -1:  # 没有"_"的情况
                root, quality = chord_value, 'None'
            else:
                root = chord_value[:split_pos]
                quality = chord_value[split_pos+1:]
            
            # 直接扩展新数据，避免多次append
            new_data.extend([
                {'name': 'Chord_Root', 'value': root},
                {'name': 'Chord_Quality', 'value': quality}
            ])
        else:
            new_data.append(item)
    
    return new_data

def modify_from_pkl(file_path, output_path, track_name='Full', mode='chord', pos=2):
    """
    从 .pkl 文件中提取和弦信息，仅对 Track 为 'Full' 的层级提取相应事件，并保存到新文件。

    :param file_path: 输入的 .pkl 文件路径
    :param output_path: 输出的 .pkl 文件路径
    :param track_name: 要提取的 Track 名称，默认为 'Full'
    :param mode: 提取模式，默认为 'chord'，可选 'melody'，'performance', 'c2m, 'm2p'
    :param pos: 要提取的事件在数据中的位置，默认为 2
    """
    # 从 .pkl 文件中加载数据
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    name_value_list = data[pos]

    # 创建一个新的列表，用于存储符合条件的数据
    new_data = []
    # 将前三行直接加入新数据
    new_data.extend(name_value_list[:3])
    # print('[debug] new_data[0]:', new_data[0])
    # Some official HookTheory files are single-track sequences without Track
    # markers. In that case the sequence after the metadata prefix is already
    # the requested track.
    body = name_value_list[3:]
    has_track_markers = any(item['name'] == 'Track' for item in body)
    if not has_track_markers:
        new_data.extend(body)
    else:
        append_flag = False
        for item in body:
            if item['name'] == 'Track':
                append_flag = item['value'] == track_name
                continue
            if append_flag:
                new_data.append(item)
    
    new_data = ensure_beat_values(new_data)
    new_data = split_chord_into_root_quality(new_data)
    
    if mode == 'performance':
        pass
    elif mode == 'chord':
        # 仅提取和弦信息
        new_data = get_chord_track(new_data)
    elif mode =='melody':
        # 仅提取旋律信息
        new_data = get_melody_track(new_data)
    elif mode == 'c2m':
        # 获取和弦和旋律数据
        chord_data = get_chord_track(new_data.copy(), num_emotions=4)
        melody_data = get_melody_track(new_data.copy())
        
        # 获取Bar索引
        chord_bars = get_bar_indices(chord_data)
        melody_bars = get_bar_indices(melody_data)
        
        # 创建合并后的数据
        merged_data = []
        
        # 处理第一个Bar之前的内容
        pre_bar_content = chord_data[:chord_bars[0]] if chord_bars else chord_data
        merged_data.extend(pre_bar_content)
        
        # 交替合并Bar部分
        min_bars = min(len(chord_bars), len(melody_bars))
        
        for i in range(min_bars):
            # 获取当前Bar范围
            chord_start = chord_bars[i]
            chord_end = chord_bars[i+1] if i+1 < len(chord_bars) else len(chord_data)
            
            melody_start = melody_bars[i]
            melody_end = melody_bars[i+1] if i+1 < len(melody_bars) else len(melody_data)
            
            # 添加和弦部分
            merged_data.append({'name': 'Track', 'value': 'Chord'})
            merged_data.extend(chord_data[chord_start:chord_end])
            
            # 添加旋律部分
            merged_data.append({'name': 'Track', 'value': 'Melody'})
            merged_data.extend(melody_data[melody_start:melody_end])
        
        new_data = merged_data
    elif mode == 'm2p':
    # 获取旋律和演奏数据
        melody_data = get_melody_track(new_data.copy())
        performance_data = new_data.copy()
        
        # 获取Bar索引
        melody_bars = get_bar_indices(melody_data)
        performance_bars = get_bar_indices(performance_data)
        
        # 创建合并后的数据
        merged_data = []
        
        # 处理第一个Bar之前的内容(取旋律数据)
        pre_bar_content = melody_data[:melody_bars[0]] if melody_bars else melody_data
        merged_data.extend(pre_bar_content)
        
        # 交替合并Bar部分
        min_bars = min(len(melody_bars), len(performance_bars))
        
        for i in range(min_bars):
            # 获取当前Bar范围
            melody_start = melody_bars[i]
            melody_end = melody_bars[i+1] if i+1 < len(melody_bars) else len(melody_data)
            
            performance_start = performance_bars[i]
            performance_end = performance_bars[i+1] if i+1 < len(performance_bars) else len(performance_data)
            
            # 添加旋律部分
            merged_data.append({'name': 'Track', 'value': 'Melody'})
            merged_data.extend(melody_data[melody_start:melody_end])
            
            # 添加演奏部分
            merged_data.append({'name': 'Track', 'value': 'Performance'})
            merged_data.extend(performance_data[performance_start:performance_end])
        
        new_data = merged_data
    # show_data(new_data, 10000)

    if new_data[-1]['name'] != 'EOS':
        new_data.append({'name': 'EOS', 'value': None})

    with open(output_path, 'wb') as f:
        pickle.dump([*find_idx(new_data, mode=mode), new_data], f)

# 找出两个数据源中所有Bar的索引位置
def get_bar_indices(data):
    return [i for i, item in enumerate(data) if item['name'] == 'Bar']

def get_chord_track(data, num_emotions=2):
    data = [item for item in data if item['name'] in ['Emotion', 'Key', 'Bar'] \
            or item['name'].startswith('Chord_') \
            or (item['name'] == 'Beat' and item['value'] in [0, 4, 8, 12])
            ]
    
    if num_emotions == 2:
        for item in data:
            if item['name'] == 'Emotion':
                if item['value'] == 'Q1' or item['value'] == 'Q4':
                    item['value'] = 'Positive'
                elif item['value'] == 'Q2' or item['value'] == 'Q3':
                    item['value'] = 'Negative'
                break
    else:
        if num_emotions != 4:
            raise ValueError("num_emotions must be 2 or 4")
        
    return data

def get_melody_track(data):
    data = [item for item in data if item['name'] in \
            ['Emotion', 'Key', 'Bar', 'Beat', 'Chord_Root', 'Chord_Quality','Note_Octave', 'Note_Degree']]
    return data

def modify_from_pkl_dir(input_dir, output_dir, track_name='Full', mode='chord', pos=2):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in tqdm(os.listdir(input_dir)):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            modify_from_pkl(file_path, output_path, track_name, mode, pos)



# file_path = r'events_temp\stage2\emopia_events\full_song_chord11_functional\events\Q1__8v0MFBZoco_0.pkl'  # 原始文件路径
# output_path = r'M.pkl'  # 输出文件路径
# # 调用函数
# pkl2leadsheet(file_path, output_path)
# print(f"处理完成，结果已保存到 {output_path}")

# file_path = r'events_temp\stage2\emopia_events\full_song_chord11_functional\events\Q1__8v0MFBZoco_0.pkl'  # 原始文件路径
# output_path = r'melody.pkl'  # 输出文件路径
# pkl2melody(file_path, output_path)

# file_path = r'events_temp\stage2\emopia_events\full_song_chord11_functional\events\Q1__8v0MFBZoco_0.pkl'  # 原始文件路径
# output_path = r'chord.pkl'  # 输出文件路径
# pkl2chord(file_path, output_path)

# file_dir = r'new_events\stage2\emopia_events\full_song_chord11_functional\events'
# # output_dir = r'new_events\melody\emopia_events_full_song_chord11_functional\events'
# # process_folder_to_melody(file_dir, output_dir)
# output_dir = r'new_events\chord\emopia_events_full_song_chord11_functional\events'
# process_folder_to_chord(file_dir, output_dir, check_track=True)

# file_dir = r'new_events\stage2\emopia_events\full_song_chord11_remi\events'
# output_dir = r'new_events\melody_remi\events'
# process_folder_to_melody(file_dir, output_dir)
# output_dir = r'new_events\chord_remi\events'
# process_folder_to_chord(file_dir, output_dir)

# file_dir = r'new_events\stage1\hooktheory_events\lead_sheet_chord11_functional\events'
# # output_dir = r'new_events\melody\hooktheory_events_lead_sheet_chord11_functional\events'
# # process_folder_to_melody(file_dir, output_dir, pos=1)
# output_dir = r'new_events\chord\hooktheory_events_lead_sheet_chord11_functional\events'
# process_folder_to_chord(file_dir, output_dir, pos=1, check_track=False)

# file_dir = r'new_events\stage2\pop1k7_events\full_song_chorder_functional\events'
# output_dir = r'new_events\chord\pop1k7_events_chord_functional\events'
# process_folder_to_chord(file_dir, output_dir, check_track=True)

# file_path = r'new_events\stage2\pop1k7_events\full_song_chorder_functional\events'
# output_path = r'new_events\chord\pop1k7_events_chord_functional\events'
# modify_from_pkl_dir(file_path, output_path, track_name='Full', mode='chord', pos=2)

# file_path = r'new_events\stage2\pop1k7_events\full_song_chorder_functional\events'
# output_path = r'new_events\melody\pop1k7_events_melody_functional\events'
# modify_from_pkl_dir(file_path, output_path, track_name='Full', mode='melody', pos=2)

# file_path = r'new_events\stage2\pop1k7_events\full_song_chorder_functional\events'
# output_path = r'new_events\c2m\pop1k7_events_c2m_functional\events'
# modify_from_pkl_dir(file_path, output_path, track_name='Full', mode='c2m', pos=2)

# file_path = r'new_events\stage2\pop1k7_events\full_song_chorder_functional\events'
# output_path = r'new_events\m2p\pop1k7_events_m2p_functional\events'
# modify_from_pkl_dir(file_path, output_path, track_name='Full', mode='m2p', pos=2)

# file_path = r'new_events\stage2\emopia_events\full_song_chord11_functional\events'
# output_path = r'new_events\c2m\emopia_events_full_song_chord11_functional\events'
# modify_from_pkl_dir(file_path, output_path, track_name='Full', mode='c2m', pos=2)

# file_path = r'new_events\stage2\emopia_events\full_song_chord11_functional\events'
# output_path = r'new_events\m2p\emopia_events_full_song_chord11_functional\events'
# modify_from_pkl_dir(file_path, output_path, track_name='Full', mode='m2p', pos=2)
