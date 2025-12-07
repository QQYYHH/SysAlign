import os
import torch
import numpy as np
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from layer import ContrastiveLearning
from utils import MyDataset, collate_fn

import matplotlib.pyplot as plt
import seaborn as sns

data_path = 'data/data.pkl'
fig_save_dir = 'picture/'

writer_idx = 0
writer = SummaryWriter('run/')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

if not os.path.exists(fig_save_dir):
    os.makedirs(fig_save_dir)

def prepare_data(data_path, train_rate):
    dataset = MyDataset(data_path)
    train_size = int(train_rate * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(5))
    return train_dataset, test_dataset, dataset

def visual_kde(data1, data2, savepath):
    sns.set(style="whitegrid")

    plt.figure(figsize=(12, 6))

    sns.kdeplot(data1, shade=True, color="red", label="positive pairs")

    sns.kdeplot(data2, shade=True, color="blue", label="negative pairs")

    # plt.title("Distance Distribution of Two Tensors", fontsize=16)
    plt.xlabel("Distance", fontsize=16)
    plt.ylabel("Density", fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(fontsize=16)
    plt.savefig(savepath)

def visual_colmatrix(c, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(c, annot=False, cmap="Blues", vmin=0, vmax=1, linewidths=0.5, square=True, cbar=True)

    plt.title("Feature Correlation Matrix (C)")
    plt.savefig(save_path)
    plt.close()

def train_epoch(dataset, model, optimizer):
    dataloader = DataLoader(dataset, 1, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    for syscall_emb, path_emb in dataloader:
        
        syscall_emb = syscall_emb.to(device)
        path_emb = path_emb.to(device)
        
        l, _, on_diag, off_diag = model(path_emb, syscall_emb, verbose=True) 

        if torch.isnan(l):
            print('NaN Loss')
            # torch.save(model.state_dict(), "model.pth")
            exit(-1)

        global writer_idx
        writer.add_scalar('Loss/train', l, writer_idx)
        writer.add_scalar('Diag/On', on_diag, writer_idx)
        writer.add_scalar('Diag/Off', off_diag, writer_idx)
        writer_idx += 1

        optimizer.zero_grad()
        l.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        writer.add_scalar("Gradient/Norm", total_norm, writer_idx)

        # scheduler.step(l)
    
def test_epoch(dataset, model, epoch):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    loss_list, c_list = [], []

    with torch.no_grad():
        for syscall_emb, path_emb in dataloader:
            syscall_emb = syscall_emb.to(device)
            path_emb = path_emb.to(device)

            l, c = model(path_emb, syscall_emb)
            loss_list.append(l.detach().cpu().item())
            c_list.append(c.detach().cpu().numpy())
            
    
    c = np.mean(c_list, axis=0)
    l = np.mean(loss_list)
    writer.add_scalar('Loss/test', l, epoch)
    save_path = os.path.join(fig_save_dir, f'{epoch}.png')
    visual_colmatrix(c, save_path)

    save_c_path = os.path.join(fig_save_dir, f'{epoch}.npy')
    np.save(save_c_path, c)

    model_save_path = os.path.join(fig_save_dir, f'{epoch}.pth')
    torch.save(model.state_dict(), model_save_path)

if __name__ == '__main__':

    train_dataset, test_dataset, dataset = prepare_data(data_path, 0.8)

    print(f'dataset loading done, and test log size is {len(test_dataset)}')

    # 定义模型、Loss、优化器
    path_input_size = 64
    path_hid_list = [64, 32] # for TCN
    syscall_input_size = 768
    sycall_hid_size = 64
    feed_size = 128
    nheads = 4
    emb_dims = [128, 64, 32]

    lr = 0.0001
    weight_decay = 0.002
    epoch = 200

    model = ContrastiveLearning(path_input_size, path_hid_list, syscall_input_size, sycall_hid_size, feed_size, nheads, emb_dims)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    scheduler = None

    for e in range(epoch):
        print(f'start epoch {e}')
        train_epoch(train_dataset, model, optimizer)
        test_epoch(test_dataset, model, e)
