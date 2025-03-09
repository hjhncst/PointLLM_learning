from collections import OrderedDict, defaultdict

import transformers
from pointllm import conversation as conversation_lib
from dataclasses import dataclass
from typing import Optional, Dict, Sequence
import torch

import numpy as np
import os

IGNORE_INDEX = -100

# * Sample Usage:
# * from utils import LRUCache
# * cache = LRUCache(capacity, max_access_count)
# if self.cache is None:
#     info_data = self.multiview_scannet[info_index]
# else:
#     info_data = self.cache.get(info_index)
#     if info_data is None or self.cache.get_access_count(info_index) >= self.cache.max_access_count:
#         # If not in cache, or accessed max_access_count times, load it and put it in cache
#         info_data = self.multiview_scannet[info_index]
#         self.cache.put(info_index, info_data)
#         self.cache.reset_access_count(info_index)

class LRUCache:
    def __init__(self, capacity, max_access_count):

        # 初始化LRU缓存，设置容量和最大访问次数
        self.cache = OrderedDict()  # 使用有序字典存储缓存数据，保持插入顺序
        self.access_count = defaultdict(int)  # 使用默认字典存储每个键的访问次数，默认值为0
        self.capacity = capacity  # 缓存的最大容量
        self.max_access_count = max_access_count  # 键的最大访问次数

    def get(self, key):
        # 获取缓存中的值，如果键不存在则返回None
        if key not in self.cache:
            return None
        value = self.cache.pop(key)  # 移除键并获取其值
        self.cache[key] = value  # Put key as the newest one
        self.access_count[key] += 1
        return value

    def put(self, key, value):
        if key in self.cache:  # Update the value and put it as newest
            self.cache.pop(key)
        elif len(self.cache) == self.capacity:  # If cache is full
            oldest_key = next(iter(self.cache))
            self.cache.popitem(last=False)  # Remove oldest item
            del self.access_count[oldest_key]  # Remove the corresponding access count
        self.cache[key] = value
        self.access_count[key] = 1

    def get_access_count(self, key):
        # 使用字典的get方法获取访问次数
        # 如果key存在于字典中，则返回对应的访问次数
        # 如果key不存在于字典中，则返回默认值0
        return self.access_count.get(key, 0)

    def reset_access_count(self, key):
        # 将指定键的访问计数重置为0
        self.access_count[key] = 0


# 预处理对话数据
def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # 复制默认的对话模板
    conv = conversation_lib.default_conversation.copy()
    # 定义角色映射，将"human"映射到对话模板的第一个角色，"gpt"映射到第二个角色
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2: # * can handle padded tokens
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX # * this is necessary for padded tokens

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len: # * unk tokens in the dialogue will cause this.
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

# 预处理多模态点云数据
def preprocess_multimodal_point_cloud(
    sources: Sequence[str],  # 输入的源数据，类型为字符串序列
    point_backbone_config: dict,  # 配置字典，包含点云处理的相关参数
    point_indicator: str = "<point>",  # 用于标识点云的占位符，默认为"<point>"
) -> Dict:  # 返回值类型为字典
    point_token_len = point_backbone_config['point_token_len']  # 从配置中获取点云令牌的长度
    default_point_patch_token = point_backbone_config['default_point_patch_token']  # 获取默认的点云补丁令牌

    for source in sources:  # 遍历所有源数据
        for sentence in source:  # 遍历源数据中的每个句子
            replace_token = default_point_patch_token * point_token_len   # 生成替换令牌，长度为点云令牌长度
            if point_backbone_config['mm_use_point_start_end']:  # 如果配置中要求使用点云的起始和结束令牌
                replace_token = point_backbone_config['default_point_start_token']+ replace_token + point_backbone_config['default_point_end_token']  # 在替换令牌前后添加起始和结束令牌
            sentence["value"] = sentence["value"].replace(point_indicator, replace_token)  # 将句子中的点云占位符替换为生成的替换令牌

    return sources  # 返回处理后的源数据

def pc_norm(pc):
    """ pc: NxC, return NxC """
    xyz = pc[:, :3]
    other_feature = pc[:, 3:]

    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / m

    pc = np.concatenate((xyz, other_feature), axis=1)
    return pc

# 用于加载和处理点云数据
def load_objaverse_point_cloud(data_path, object_id, pointnum=8192, use_color=False):
    # 构建文件名，格式为 "object_id_pointnum.npy"
    filename = f"{object_id}_{pointnum}.npy"
    point_cloud = np.load(os.path.join(data_path, filename))

    # * normalize
    point_cloud = pc_norm(point_cloud)

    if not use_color:
        point_cloud = point_cloud[:, :3]

    return point_cloud


# 定义一个数据集类，用于加载和处理点云数据
@dataclass
class DataCollatorForPointTextDataset(object):
    """Collate examples for mixed dataset with text and point cloud data."""

    tokenizer: transformers.PreTrainedTokenizer

    # 定义一个tokenizer属性，用于处理文本数据
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 定义一个可调用方法，用于将输入的实例列表整理成批量数据
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        # 从输入的实例列表中提取input_ids和labels，分别存储在两个列表中
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        # 使用pad_sequence函数对input_ids进行填充，使其长度一致，填充值为tokenizer的pad_token_id
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        # 使用pad_sequence函数对labels进行填充，使其长度一致，填充值为IGNORE_INDEX
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # 创建一个字典，包含填充后的input_ids、labels和attention_mask
        # attention_mask用于指示哪些位置是有效的（非填充位置）
        if 'point_clouds' in instances[0]:
            # 检查第一个实例中是否包含'point_clouds'键
            point_clouds = [instance['point_clouds'] for instance in instances]
            if all(x is not None and x.shape == point_clouds[0].shape for x in point_clouds): # * point_clouds have different shapes
                batch['point_clouds'] = torch.stack(point_clouds)
            else:
                batch['point_clouds'] = point_clouds # * return as lists

        return batch

# 定义最远采样点云函数（降维）
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

# 点云数据标准化
def pc_normalize(pc):
    """
    pc: Nx3 array
    This functions normalizes a point cloud to fit within a unit sphere.
    It first calculates the centroid of the point cloud and then subtracts
    it from all points before scaling all points to fit within a unit sphere.
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
