import torch
from torch import nn

from sources.word_embedding import WordEmbedding


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.word2vec = WordEmbedding()
        self.get_vector = self.word2vec.get_vector
        self.gru = nn.GRU(input_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_word, hidden=None):
        embedded = self.make_embedding(input_word)
        outputs, hidden = self.gru(embedded, hidden)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        return outputs, hidden

    def make_embedding(self, input_word):
        tokens = input_word.split()
        token_len = len(tokens)
        word_embedding = torch.zeros([token_len, self.input_size])
        for i in range(token_len):
            vector = self.get_vector(tokens[i])
            word_embedding[i, :] = torch.from_numpy(vector)
        word_embedding = word_embedding.unsqueeze(0)
        return word_embedding.to(self.device)


# input_str = 'railways was asked by a court to pay number compensation to a family'
# en = Encoder(300,128)
# en.to(en.device)
# a = en(input_str)
# print(a)

# class EncoderRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, is_bidirect, num_layers=2, dropout=0.3):
#         super(EncoderRNN, self).__init__()
#
#         self.n_layers = num_layers
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.dropout = dropout
#
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             bidirectional=True,
#             dropout=dropout if num_layers > 1 else 0,
#             batch_first=True
#         )
#
#     def forward(self, input, hidden=None):
#         # Convert word indexes to embeddings
#         embedded = self.make_embedding(input)
#         outputs, hidden = self.lstm(embedded, hidden)
#         # Sum bidirectional GRU outputs
#         outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
#         # Return output and final hidden state
#         return outputs, hidden
#
#     def make_embedding(self, input):
#         tokens = input.split()
#         token_len = len(tokens)
#         word_embedding = torch.zeros([token_len, self.input_size])
#         for i in range(token_len):
#             vector = self.get_vector(tokens[i])
#             word_embedding[i, :] = torch.from_numpy(vector)
#         word_embedding = word_embedding.unsqueeze(0)
#         return word_embedding.to(self.device)