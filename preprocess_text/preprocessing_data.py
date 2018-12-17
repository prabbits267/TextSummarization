import operator
import nltk
import pandas as pd
from config import *


class DataPreprocess:
    def __init__(self):
        self.resources_path = RESOURCES_PATH
        self.content, self.summary, self.tokens = self.read_file()


    def read_file(self):
        d_frame = pd.read_csv(self.resources_path, skiprows=1, encoding='latin-1')
        content = d_frame.iloc[:, 5]
        summary = d_frame.iloc[:, 4]
        tokens = [str(w) for w in list(summary)]
        tokens = ' '.join(list(tokens))
        tokens = nltk.word_tokenize(tokens)
        return list(content), list(summary), tokens

    def token_statictics(self):
        word_count = dict()
        for token in self.tokens:
            if token not in word_count:
                word_count[token] = 1
            else:
                word_count[token] += 1
        return word_count

    def write_wcount(self, word_count):
        word_count = sorted(word_count.items(), key=operator.itemgetter(1))
        with open(WORD_COUNT, 'wt') as file_writer:
            for pair in word_count:
                file_writer.write(pair[0] + '\t\t\t' + str(pair[1]) + '\n')
                # file_writer.write(pair[0] + '\n')

data_pre = DataPreprocess()
wc = data_pre.token_statictics()
data_pre.write_wcount(wc)