import torch
import pickle
import csv
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def numpy_to_tensor(arr, use_gpu=True):
    if use_gpu:
        return torch.tensor(arr).to(device).float()
    else:
        return torch.tensor(arr).float()


def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def list2str(a_list):
    return ''.join([str(i) for i in a_list])


def pickle_load(f):
    return pickle.load(open(f, 'rb'))


def pickle_dump(obj, f):
    pickle.dump(obj, open(f, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def json_read(path):
    with open(path, 'r') as f:
        content = json.load(f)
    f.close()
    return content


def json_write(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)
    f.close()


def csv_read(path):
    content = list()
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            content.append(row)
    f.close()
    header = content[0]
    content = content[1:]
    return header, content

def extract_prefixes(data):
    """
    提取数据中所有不同的前缀（如 Key_, Chord_Quality_, 等）
    基于最后一个下划线的位置提取前缀。
    """
    prefixes = set()
    for key in data.keys():
        # 找到最后一个下划线的位置
        underscore_index = key.rfind('_')
        if underscore_index != -1:
            # 提取前缀（包括最后一个下划线）
            prefix = key[:underscore_index + 1]
            prefixes.add(prefix)
    return sorted(list(prefixes))  # 返回排序后的列表

def count_by_prefix(data):
    """
    统计每个前缀对应的键的数量
    基于最后一个下划线的位置提取前缀。
    """
    # 提取所有前缀
    prefixes = extract_prefixes(data)
    
    # 初始化统计字典
    prefix_counts = {prefix: 0 for prefix in prefixes}
    
    # 遍历数据，统计每个前缀的数量
    for key in data.keys():
        # 找到最后一个下划线的位置
        underscore_index = key.rfind('_')
        if underscore_index != -1:
            # 提取前缀
            prefix = key[:underscore_index + 1]
            if prefix in prefix_counts:
                prefix_counts[prefix] += 1
    
    return prefix_counts

def show_prefix_counts(data):
    if isinstance(data, tuple) and len(data) > 0:
        data = data[0]  
    # 统计前缀数量
    prefix_counts = count_by_prefix(data)
    
    # 打印统计结果
    print('Prefix counts:')
    for prefix, count in prefix_counts.items():
        print(f'{prefix}: {count}')

def get_value(data, mask, prefix_list, exclude=False):
    """
    根据前缀将数据分类到三个列表中
    :param data: 输入的字典数据
    :param prefix_list: 需要提取到listA的前缀列表
    :param mask: 需要单独提取到mask的前缀
    :return: listA, listB, v_mask
    """
    if isinstance(data, tuple) and len(data) > 0:
        data = data[0] 

    listA = []
    listB = []
    v_mask = None
    
    for key in data:
        # 提取前缀，方法与count_by_prefix一致
        underscore_index = key.rfind('_')
        if underscore_index == -1:
            continue  # 忽略没有下划线的键
        prefix = key[:underscore_index + 1]
        value = data[key]
        
        # 检查是否是mask的前缀
        if prefix == mask:
            v_mask = value
        
        # 检查是否在prefix_list中
        if prefix in prefix_list:
            listA.append(value)
        else:
            # 不在prefix_list中且不是mask的前缀，加入listB
            if prefix != mask:
                listB.append(value)
    if exclude:
        return listA, listB, v_mask
    else:
        return listB, listA, v_mask

