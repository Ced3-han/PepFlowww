import math
import torch
from torch.utils.data._utils.collate import default_collate
from pepflow.modules.protein.constants import PAD_RESIDUE_INDEX

import os


DEFAULT_PAD_VALUES = {
    'aa': PAD_RESIDUE_INDEX, #0-20,+21
    'chain_id': ' ', 
    'icode': ' ',
}

DEFAULT_NO_PADDING = {
    # 'origin',
}

class PaddingCollate(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, no_padding=DEFAULT_NO_PADDING, eight=True):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.no_padding = no_padding
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys


    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        keys = self._get_common_keys(data_list)
        
        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        for data in data_list:
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_pad_value(k)) if k not in self.no_padding else v
                for k, v in data.items()
                if k in keys
            }
            data_padded['res_mask'] = (self._get_pad_mask(data[self.length_ref_key].size(0), max_length))
            data_list_padded.append(data_padded)
        return default_collate(data_list_padded)


def apply_patch_to_tensor(x_full, x_patch, patch_idx):
    """
    Args:
        x_full:  (N, ...)
        x_patch: (M, ...)
        patch_idx:  (M, )
    Returns:
        (N, ...)
    """
    x_full = x_full.clone()
    x_full[patch_idx] = x_patch
    return x_full


def index_select(v, index, n):
    if isinstance(v, torch.Tensor) and v.size(0) == n:
        return v[index]
    elif isinstance(v, list) and len(v) == n:
        return [v[i] for i in index]
    else:
        return v


def index_select_data(data, index):
    return {
        k: index_select(v, index, data['aa'].size(0))
        for k, v in data.items()
    }


def mask_select(v, mask):
    if isinstance(v, torch.Tensor) and v.size(0) == mask.size(0):
        return v[mask]
    elif isinstance(v, list) and len(v) == mask.size(0):
        return [v[i] for i, b in enumerate(mask) if b]
    else:
        return v


def mask_select_data(data, mask):
    return {
        k: mask_select(v, mask)
        for k, v in data.items()
    }


def find_longest_true_segment(input_tensor):
    max_segment_length = 0
    max_segment_start = 0
    current_segment_length = 0
    current_segment_start = 0
    input_list = input_tensor.tolist()  # 转换为Python列表以便遍历

    for i, value in enumerate(input_list):
        if value:  # 如果当前位置为True
            current_segment_length += 1
            if current_segment_length > max_segment_length:
                max_segment_length = current_segment_length
                max_segment_start = current_segment_start
        else:
            current_segment_length = 0
            current_segment_start = i + 1

    # 创建一个新的PyTorch Tensor，将最长的True段位置置为True，其他位置置为False
    result_tensor = torch.zeros_like(input_tensor, dtype=torch.bool)
    result_tensor[max_segment_start:max_segment_start + max_segment_length] = True

    return result_tensor

def get_test_batch(dataset_dir='/datapool/data2/home/jiahan/Res Proj/PepDiff/PepFlow/Data',name='batch.pt'):
    return torch.load(os.path.join(dataset_dir,name))

if __name__ == '__main__':
    batch = get_test_batch()
    print(batch)

