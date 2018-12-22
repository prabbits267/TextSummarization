import torch
from torch import nn
import torch.nn.functional as F
import time as t
# input : decoder_output, current hidden_state
# # output : attn energy (seq_len)
# class Attn(nn.Module):
#     def __init__(self, method, hidden_size):
#         super(Attn, self).__init__()
#         self.method = method
#         self.hidden_size = hidden_size
#         if self.method == 'general':
#             self.attn = nn.Linear(self.hidden_size, hidden_size)
#         elif self.method == 'concat':
#             self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
#             self.v = nn.Parameter(torch.FloatTensor(hidden_size))
#         self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#
#     # single sentence ht, hs
#     # return tensor(sequence len) attention paid to encoder input
#     def forward(self, hidden, encoder_output):
#         att_time = t.time()
#         hidden = hidden.squeeze(0).squeeze(0)
#         encoder_output = encoder_output.squeeze(0)
#         hidden = hidden.to(self.device)
#         encoder_output = encoder_output.to(self.device)
#         seq_len = len(encoder_output)
#         attn_energies = torch.zeros(seq_len).to(self.device)
#         total = 0
#         for i in range(seq_len):
#             time = t.time()
#             hd = hidden
#             attn_energies[i] = self.score(hidden, encoder_output[i])
#             execute = (t.time() - time)
#             total += execute
#         attn_energies = F.softmax(attn_energies, dim=0)
#         return attn_energies
#
#     # hidden (batch, 1, hidden_size) (batch, time_step, hidden_size)
#     def score(self, hidden, encoder_output):
#         if self.method == 'dot':
#             energy = torch.dot(hidden.view(-1), encoder_output.view(-1))
#             return energy
#         elif self.method == 'general':
#             energy = self.attn(encoder_output)
#             energy = torch.dot(hidden, energy)
#             return energy
#         elif self.method == 'concat':
#             energy = self.attn(torch.cat((hidden[0], encoder_output), 0))
#             energy = self.v.dot(energy)
#             return energy

class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))
        self.softmax = nn.LogSoftmax(dim=1)

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        return self.softmax(attn_energies)

