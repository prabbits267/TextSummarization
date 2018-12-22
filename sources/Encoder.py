import torch
from torch import nn
from torch.autograd import Variable
from sources.word_embedding import WordEmbedding

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, is_bidirect, num_layers=2):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_bidirect = is_bidirect
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=is_bidirect,
                            dropout=0.2,
                            batch_first=True)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.word2vec = WordEmbedding()
        self.get_vector = self.word2vec.get_vector

    def forward(self, input):
        word_embedding = self.make_embedding(input)
        output, (hidden_state, cell_state) = self.lstm(word_embedding, None)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return output, (hidden_state, cell_state)

    def make_embedding(self, input):
        tokens = input.split()
        token_len = len(tokens)
        word_embedding = torch.zeros([token_len, self.input_size])
        for i in range(token_len):
            vector = self.get_vector(tokens[i])
            word_embedding[i, :] = torch.from_numpy(vector)
        word_embedding = word_embedding.unsqueeze(0)
        return word_embedding.to(self.device)


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