import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import NLLLoss, CrossEntropyLoss
from torch.utils.data import DataLoader
import time as t
from sources.Decoder import Decoder
from sources.Encoder import Encoder
from sources.dataset import Seq2SeqDataset

START_OF_SENT = 'S.O.S'
END_OF_SENT = 'E.O.S'

class Train():
    def __init__(self, input_size, hidden_size, batch_size, learning_rate, num_epoch, method):
        dataset = Seq2SeqDataset()
        self.vocab = dataset.word_type
        self.vocab_size = len(self.vocab)
        self.word2ind, self.ind2word = self.gen_dict()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = self.vocab_size
        self.method = method
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.is_bidirect = True

        self.dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        self.encoder = Encoder(self.input_size, self.hidden_size, self.is_bidirect)
        self.decoder = Decoder(self.input_size, self.hidden_size, self.output_size, self.is_bidirect, self.method)

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        self.loss_function = CrossEntropyLoss()

        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optim = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)

    def step(self, input, target):
        self.encoder.train()
        self.decoder.train()

        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()

        target = target.split()
        target_tensor = self.convert2indx(target)

        encoder_output, (hidden_state, cell_state) = self.encoder(input)

        target_len = len(target) - 1

        decoder_output = torch.zeros([len(target), self.output_size]).to(self.device)
        output_index = torch.zeros(target_len)
        # use teacher forcing
        decoder_input = START_OF_SENT
        for i in range(target_len):
            output, (hidden_state, cell_state) = self.decoder(decoder_input, (hidden_state, cell_state), encoder_output)
            decoder_output[i] = output.squeeze(0)
            output_index[i] = output.topk(1)[1]
            decoder_input = target[i+1]

        decoder_output = self.create_variable(decoder_output)

        loss = self.loss_function(decoder_output, target_tensor.squeeze(0))

        loss.backward()

        _ = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 50.0)
        _ = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50.0)

        self.encoder_optim.step()
        self.decoder_optim.step()

        return loss.data[0], decoder_output, output_index

    def train_batch(self):
        total_loss = 0
        for i, (x_data, y_data) in enumerate(self.dataloader):
            loss, _, output = self.step(x_data[0], y_data[0])
            total_loss += loss
            if i % 1 == 0:
                # print(x_data)
                print('Interation %s , loss %.3f '%(i, loss.cpu().numpy()))
                # print()
        return total_loss/len(self.dataloader)



    def create_variable(self, tensor):
        return Variable(tensor.to(self.device), requires_grad=True)

    def train(self):
        for i in range(self.num_epoch):
            loss = self.train_batch()
            print('Epoch : ', i, ' -->>>--> loss', loss)

    def convert2indx(self, input):
        input_tensor = torch.LongTensor([[self.word2ind[w] for w in input]])
        return input_tensor.to(self.device)

    def gen_dict(self):
        word2ind = {w:i for i, w in enumerate(self.vocab)}
        ind2word = {w[1]:w[0] for w in word2ind.items()}
        return word2ind, ind2word

# input_size, hidden_size, batch_size, learning_rate, method
train = Train(300, 128, 1, 0.2, 50, 'general')
train.train()
