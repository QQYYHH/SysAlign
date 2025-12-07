import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pad_sequence

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        # self.relu1 = nn.ReLU()
        self.relu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation)
        
        self.chomp2 = Chomp1d(padding)
        # self.relu2 = nn.ReLU()
        self.relu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        # self.relu = nn.ReLU()
        self.relu = nn.ELU()
        self.init_weights()

    def init_weights(self):
        # nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.constant_(self.conv1.bias, 0.01)  

        # nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.constant_(self.conv1.bias, 0.01)  

        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
            # nn.init.kaiming_normal_(self.downsample.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.constant_(self.downsample.bias, 0.01) 

    def forward(self, x):
        # out = self.net(x)
        out = self.conv1(x)

        out = self.chomp1(out)

        out = self.relu1(out)

        out = self.dropout1(out)

        out = self.conv2(out)

        out = self.chomp2(out)

        out = self.relu2(out)

        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# TCN 
class PathEncoder(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(PathEncoder, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x : (N, L, C)
        x = x.transpose(1, 2)
        output = self.network(x).transpose(1, 2)
        output = output.mean(dim=1)
        return output

# Transformer
class SyscallEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, feed_size, nheads):
        super(SyscallEncoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            nn.LayerNorm(hidden_size), 
            # nn.ReLU()
            nn.ELU()
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nheads, feed_size), 
            num_layers=2
        )

    def forward(self, x):
        # (N, seq, size) -> (N, seq, size)
        x = self.fc(x)
        x = self.encoder(x)
        # (N, size)
        x = x.mean(dim=1)
        return x

class ContrastiveLearning(nn.Module):
    def __init__(self, path_input_size, path_hid_list, syscall_input_size, syscall_hid_size, feed_size, nheads, emb_dims):
        super(ContrastiveLearning, self).__init__()
        self.alpha = 0.5
        self.pathencoder = PathEncoder(path_input_size, path_hid_list)

        proj = [nn.Linear(path_hid_list[-1], emb_dims[0]), nn.LayerNorm(emb_dims[0]), nn.ELU()]
        for i in range(len(emb_dims)-2):
            proj.append(nn.Linear(emb_dims[i], emb_dims[i+1]))
            proj.append(nn.LayerNorm(emb_dims[i+1]))
            proj.append(nn.ELU())
        proj.append(nn.Linear(emb_dims[-2], emb_dims[-1]))
        self.proj1 = nn.Sequential(*proj)

        self.syscallencoder = SyscallEncoder(syscall_input_size, syscall_hid_size, feed_size, nheads)

        proj = [nn.Linear(syscall_hid_size, emb_dims[0]), nn.LayerNorm(emb_dims[0]), nn.ELU()]
        for i in range(len(emb_dims)-2):
            proj.append(nn.Linear(emb_dims[i], emb_dims[i+1]))
            proj.append(nn.LayerNorm(emb_dims[i+1]))
            proj.append(nn.ELU())
        proj.append(nn.Linear(emb_dims[-2], emb_dims[-1]))
        self.proj2 = nn.Sequential(*proj)

        self.ln = nn.LayerNorm(emb_dims[-1])

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, path_emb, syscall_emb, verbose=False):
        # (seq, size) -> (1, seq, size)
        path_emb = path_emb.unsqueeze(0)
        # (1, size)
        path_emb = self.pathencoder(path_emb)
        path_emb = self.ln(self.proj1(path_emb))
        
        # (seq, size) -> (1, seq, size)
        syscall_emb = syscall_emb.unsqueeze(0)
        # (1, size)
        syscall_emb = self.syscallencoder(syscall_emb)
        syscall_emb = self.ln(self.proj2(syscall_emb))
        
        return path_emb, syscall_emb


class DistanceLoss(nn.Module):
    def __init__(self) -> None:
        super(DistanceLoss, self).__init__()
        self.m = 2 
    
    def forward(self, src_embs, dst_embs, labels):
        d = torch.norm(src_embs-dst_embs, dim=1)
        positive_part = (1-labels) * d.pow(2)
        negative_part = labels * torch.pow(torch.clamp(self.m - d, min=0), 2)
        l = positive_part + negative_part
        return l.mean()


class SimilarityLoss(nn.Module):
    def __init__(self):
        self.alpha = 0.5
        self.label_factor = 0.25
        super(SimilarityLoss, self).__init__()

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    
    def forward(self, path_emb, syscall_emb, label, verbose=False):
        c = path_emb.T @ syscall_emb
        on_diag = torch.diagonal(c).add_(label - 1).pow_(2).sum()
        # off_diag = self.off_diagonal(c).add_(-label).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()

        loss = on_diag + self.alpha * off_diag

        return loss, c
