import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# input : decoder_output, current hidden_state
# output : attn energy (seq_len)
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # single sentence ht, hs
    # return tensor(sequence len) attention paid to encoder input
    def forward(self, hidden, encoder_output):
        hidden = hidden.squeeze(0).squeeze(0)
        encoder_output = encoder_output.squeeze(0)
        seq_len = len(encoder_output)
        attn_energies = Variable(torch.zeros(seq_len))
        attn_energies = self.create_variable(attn_energies)
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_output[i])
        return F.softmax(attn_energies, dim=0)

    def create_variable(self, tensor):
        return Variable(tensor.to(self.device))

    # hidden (batch, 1, hidden_size) (batch, time_step, hidden_size)
    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = torch.dot(hidden.view(-1), encoder_output.view(-1))
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.dot(hidden, energy)
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden[0], encoder_output), 0))
            energy = self.v.dot(energy)
            return energy