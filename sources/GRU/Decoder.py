import torch
from torch import nn
import torch.nn.functional as F
from sources.attn import Attn
from sources.word_embedding import WordEmbedding
from sources.GRU.Encoder import *
device = 'cuda:0' if torch.cuda.is_available() else "cpu"

class Decoder(nn.Module):
    def __init__(self, attn_model, input_size, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(Decoder, self).__init__()

        self.attn_model = attn_model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.word2vec = WordEmbedding()
        self.get_vector = self.word2vec.get_vector

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_word, last_hidden, encoder_outputs):
        embedded = self.make_embedding(input_word)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.mm(encoder_outputs)
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden

    def make_embedding(self, input):
        word_embed = self.get_vector(input)
        word_embed = torch.from_numpy(word_embed)
        word_embed = word_embed.unsqueeze(0).unsqueeze(0)
        return word_embed.to(self.device)

input = 'railways was asked by a court to pay number compensation to a family'
encoder = Encoder(input_size=300, hidden_size=128)
encoder.to('cuda:0')
out, hc_state = encoder(input)

decoder = Decoder('general', 300, 128, 40912, n_layers=2)
decoder.to('cuda:0')
z = decoder('S.O.S', hc_state, out)
print(z)
