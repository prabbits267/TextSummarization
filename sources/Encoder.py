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
        return output, (hidden_state, cell_state)

    def create_variable(self, tensor):
        return Variable(tensor.to(self.device))

    def make_embedding(self, input):
        tokens = input.split()
        token_len = len(tokens)
        word_embedding = torch.zeros([token_len, self.input_size])
        for i in range(token_len):
            vector = self.get_vector(tokens[i])
            word_embedding[i, :] = torch.from_numpy(vector)
        word_embedding = word_embedding.unsqueeze(0)
        return self.create_variable(word_embedding)


# input = 'railways was asked by a court to pay number compensation to a family'
# encoder = Encoder(input_size=300, hidden_size=64)
# encoder.to('cuda:0')
# a = encoder(input)
# print(a)