import os
import torch
import random
import pickle
import itertools
import numpy as np
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from layer import ContrastiveLearning

from collections import Counter

data_path = 'data/data.pkl'

random.seed(42) 

nginx_poc_logs = ['177333.177333.log']
nginx_poc_syscall_idx = {'177333.177333.log': -16}

class MyDataset(Dataset):
    def __init__(self, data_path, poclogs, poc_syscall_idx):
        with open(data_path, 'rb') as fd:
            self.data = pickle.load(fd)

        self.same_syscall_seq_threshold = 1

        self.poclogs = poclogs
        self.poc_syscall_idx = poc_syscall_idx
        self.max_path_length = 1024

        self.preprocess()
        print('preprocess done...')

        # 所有数据归一化
        self.norm()
        print('normalization done...') # 直接计算距离的话，不做归一化好一点

        k = 5
        self.syscall_emb_list, self.path_emb_list, self.labels = self.create_samples_attack(k)

    def preprocess(self):
        syscall_seq_distribute = [tuple(x) for x, _, _, _, _ in self.data]
        
        comm_prefix = [ ]
        len_comm_prefix = len(comm_prefix)
        if len_comm_prefix > 0:
            for i, item in enumerate(self.data):
                syscall_num_list = item[0]
                if syscall_num_list[: len_comm_prefix] == comm_prefix:
                    syscall_num_list = syscall_num_list[len_comm_prefix: ]
                    syscall_emb_arr, path_emb_arr, path_end_list = item[1], item[2], item[3]
                    log_name = item[4]
                    start_indices = [0] + path_end_list[:-1]  # 自动生成起始索引

                    split_arrays = [path_emb_arr[start: end] for start, end in zip(start_indices, path_end_list)]
                    assert len(syscall_emb_arr) == len(split_arrays)

                    split_arrays = split_arrays[len_comm_prefix: ]

                    path_emb_arr = np.concatenate(split_arrays, axis=0)
                    path_end_list = [0]
                    for arr in split_arrays:
                        path_end_list.append(path_end_list[-1] + len(arr))
                    path_end_list = path_end_list[1: ]

                    self.data[i] = (syscall_num_list[len_comm_prefix: ], item[1][len_comm_prefix: ], path_emb_arr, path_end_list, log_name)
        self.data = [item for item in self.data if len(item[0]) != 0]

    def normalize_and_restore(self, numpy_list, method='standard'):
        concatenated = np.concatenate(numpy_list, axis=0)  # (total_k, size)
        

        mn = np.min(concatenated, axis=0, keepdims=True)
        mx = np.max(concatenated, axis=0, keepdims=True)
        normalized = (concatenated - mn) / (mx - mn + 1e-6)  

        if np.isinf(normalized).any() or np.isnan(normalized).any():
            exit(-1)

        split_indices = [arr.shape[0] for arr in numpy_list]

        restored_list = np.split(normalized, np.cumsum(split_indices)[:-1], axis=0)

        return restored_list

    def norm(self, method='standard'):

        syscall_num_batch, syscall_emb_batch, path_emb_batch, path_end_batch, log_name_batch = zip(*self.data)
        syscall_emb_batch = self.normalize_and_restore(syscall_emb_batch)

        path_emb_batch = self.normalize_and_restore(path_emb_batch)

        self.data = tuple(zip(syscall_num_batch, syscall_emb_batch, path_emb_batch, path_end_batch, log_name_batch))


    def check_mal_log_name(self, log_name):
        for name in self.poclogs:
            if name in log_name:
                return name
        return None


    def create_samples(self):
        samples = [] # [(syscall_emb, path_emb)]

        for _, syscall_emb_arr, path_emb_arr, path_end, log_name in self.data:
            # syscall_emb_arr (K, size) ; path_emb_arr (T, size);
            assert len(syscall_emb_arr) == len(path_end)

            print(f'syscall lenght: {len(syscall_emb_arr)}')

            orig_mal_name = self.check_mal_log_name(log_name)
            if orig_mal_name:

                mal_idx = self.poc_syscall_idx[orig_mal_name]
                if mal_idx < 0:

                    mal_idx = len(syscall_emb_arr) + mal_idx

            for i, end_idx in enumerate(path_end):
                syscall_emb = syscall_emb_arr[: i+1]
                path_emb = path_emb_arr[: end_idx]


                label = 1 if orig_mal_name and i >= mal_idx else 0

                samples.append((syscall_emb, path_emb, label))

        syscall_emb_list, path_emb_list, labels = zip(*samples)
        print(Counter(labels))

        return syscall_emb_list, path_emb_list, labels

    def random_delete_sequnce_in_exepath(self, path_emb, k):
        n = path_emb.shape[0]

        start_index = np.random.randint(0, n)
        num_to_delete = max(min(k, n - start_index-1), 0)

        path_emb = np.delete(path_emb, np.s_[start_index: start_index + num_to_delete], axis=0)
        return path_emb
    
    def create_samples_attack(self, k):
        samples = []
        for syscall_num_list, syscall_emb_arr, path_emb_arr, path_end, log_name in self.data:
            assert len(syscall_emb_arr) == len(path_end)
            for i, end_idx in enumerate(path_end):
                syscall_emb = syscall_emb_arr[: i+1]
                path_emb = path_emb_arr[: end_idx]
                samples.append((syscall_emb, path_emb, 0))

                path_emb = self.random_delete_sequnce_in_exepath(path_emb, k)
                samples.append((syscall_emb, path_emb, 1))

        syscall_emb_list, path_emb_list, label_list = zip(*samples)
        print(Counter(label_list))
        return syscall_emb_list, path_emb_list, label_list

    def __len__(self):
        return len(self.syscall_emb_list)
    
    def __getitem__(self, idx):
        return self.syscall_emb_list[idx], self.path_emb_list[idx], self.labels[idx]

def collate_fn(batch):
    syscall_emb = batch[0][0]
    path_emb = batch[0][1]
    label = batch[0][2]

    syscall_emb = torch.tensor(syscall_emb, dtype=torch.float)
    path_emb = torch.tensor(path_emb, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long)

    return syscall_emb, path_emb, label

if __name__ == '__main__':

    dataset = MyDataset(data_path, nginx_poc_logs, nginx_poc_syscall_idx)
    dataloader = DataLoader(dataset, 1, shuffle=False, collate_fn=collate_fn, pin_memory=True)

    for syscall_emb, path_emb, label in dataloader:

        path_input_size = 64
        path_hid_list = [64, 32] # for TCN
        syscall_input_size = 768
        sycall_hid_size = 64
        feed_size = 128
        nheads = 4
        emb_dims = [128, 64, 32]

        model = ContrastiveLearning(path_input_size, path_hid_list, syscall_input_size, sycall_hid_size, feed_size, nheads, emb_dims)

        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.002)

        l, c = model(path_emb, syscall_emb)



