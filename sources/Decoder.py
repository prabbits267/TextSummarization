import torch
from torch import nn
import time as t

from sources.Encoder import Encoder
from sources.attn import Attn
from sources.word_embedding import WordEmbedding

device = 'cuda:0' if torch.cuda.is_available() else "cpu"

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, is_bidirect, method, num_layers=2):
        super(Decoder, self).__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.method = method
        self.is_bidirect = is_bidirect
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=is_bidirect,
            batch_first=True
        )
        self.num_dir = 2 if self.lstm.bidirectional else 1
        self.attn = Attn(method, hidden_size)
        self.attn = self.attn.to(self.device)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.word2vec = WordEmbedding()
        self.get_vector = self.word2vec.get_vector

    # input ([[1]]) hc_state ((1,1,64), (1,1,64)) encoder_output (1,<time_step>,64)
    def forward(self, input, hc_state, encoder_output):
        word_embed = self.make_embedding(input)
        lstm_output, (hidden_state, cell_state) = self.lstm(word_embed, hc_state)
        # (1, 1, 64) , encoder_output (1, <time_step>, 64) ==> (<time_step>)
        lstm_output = lstm_output[:, :, :self.hidden_size] + lstm_output[:, :, self.hidden_size:]
        attn_weight = self.attn(lstm_output, encoder_output)
        # print(attn_weight)
        # attn_weight (1, time_step) encoder_output(time_step, hidden_size)  ==> context: (1, hidden_size)
        # rest_time = t.time()
        encoder_output = encoder_output.squeeze(0)
        context = attn_weight.mm(encoder_output)
        # context: (1, hidden_size)  hidden_state (1, 64) ==> (1, 64)
        concat_input = torch.cat((lstm_output.squeeze(0), context), 1)
        attention_hs = self.tanh(self.concat(concat_input))
        output = self.out(attention_hs)
        output = self.softmax(output)

        return output, (hidden_state, cell_state)

    def make_embedding(self, input):
        word_embed = self.get_vector(input)
        word_embed = torch.from_numpy(word_embed)
        word_embed = word_embed.unsqueeze(0).unsqueeze(0)
        return word_embed.to(self.device)

    def init_hc_state(self):
        hidden_state = torch.zeros([self.num_layers * self.num_dir, 1, self.hidden_size])
        cell_state = hidden_state
        return cell_state.to(self.device), hidden_state.to(self.device)

input = 'railways was asked by a court to pay number compensation to a family'
encoder = Encoder(input_size=300, hidden_size=128, is_bidirect=True)
encoder.to('cuda:0')
out, hc_state = encoder(input)

# decoder = Decoder(300, 128, 40912, True, 'general')
# decoder.to('cuda:0')
# hc_state = decoder.init_hc_state()
# z = decoder('S.O.S', hc_state, out)
# print(z)
