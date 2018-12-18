import operator
import nltk
import pandas as pd
from nltk import word_tokenize

from config import *
import re

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

    # return dictionary word_count {word:count}
    def token_statictics(self, tokens):
        word_count = dict()
        for token in tokens:
            if token not in word_count:
                word_count[token] = 1
            else:
                word_count[token] += 1
        return word_count

    # write word_count into file
    def write_wcount(self, word_count):
        word_count = sorted(word_count.items(), key=operator.itemgetter(1))
        with open(WORD_COUNT, 'wt') as file_writer:
            for pair in word_count:
                file_writer.write(pair[0] + '\t\t\t' + str(pair[1]) + '\n')

    def write_content(self):
        with open(CONTENT_PATH, 'wt', encoding='utf-8') as file_writer:
            content = [str(w) for w in self.content]
            for cont in content:
                file_writer.write(cont + '\n')

    def preprocess_paragraph(self, para):
        para = re.sub('Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday', 'DAYOFWEEK', para)
        para = re.sub('\(HT\sPhoto\)|\(HT\?FILE\)|\(HT\?File\)|\(HT File\)|\(Picture: Sourced\)|\(HTPhoto\)', '', para)
        # para = re.sub('(ALSO\sREAD.*$)|(Also\sRead)|(also\sRead)', '', para)
        para = re.sub('(ALSO READ.*)|(Also Read.*)|(also Read.*)|(Also read.*)', '', para)
        para = re.sub('\d+\.\d{2}\s?(pm|am)?', 'HOUR', para)
        para = re.sub('\d+:\d{2}(am|pm)', 'HOUR', para)
        para = re.sub('January|February|March|April|May|June|Junly'
                      '|August|September|October|November|December', 'MONTH', para)
        para = re.sub('#\w+', 'HASHTAG', para)
        para = re.sub('^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)'
                      '|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])'
                      '|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$',
                      'EMAIL', para)
        para = re.sub('(\d+)(-\d+){1,}', 'NUMBER', para)
        para = re.sub('^((http[s]?|ftp):\/)?\/?([^:\/\s]+)((\/\w+)*\/)'
                      '([\w\-\.]+[^#?\s]+)(.*)?(#[\w\-]+)?$', 'URL', para)
        para = re.sub('pic.twitter.com\/\w+|twitter.com/\w+', 'URL', para)
        para = re.sub('https?:[^\s]+', 'URL', para)
        para = re.sub('\w+\.twitter.com/\w+', 'URL', para)
        para = re.sub('\{"@context.*}}', '', para)
        para = re.sub("([a-zA-Z]+)?\*+([a-zA-Z])?", 'F-WORD', para)
        para = re.sub('fakenote_boxwrap.*differenceNUM\.', '', para)
        para = re.sub('\?|"|\.{2,}', " ", para)

        return para

    # adding white space to paragraph ago.Sources ==> ago. Sources
    def adding_wspace(self, para):
        fiall = re.findall('[a-zA-Z]+\.\[a-zA-Z]+', para)
        if fiall:
            for fi in fiall:
                ind = para.index(fi)
                stop_ind = fi.index('.') + 1
                ind += stop_ind
                para = para[:ind] + ' ' + para[ind:]
        return para

    def write_csv_file(self):
        cont_summr = list()
        for i, cont in enumerate(self.content):
            if cont and self.summary[i]:
                cont = self.preprocess_paragraph(cont)
                sum = self.preprocess_paragraph(self.summary[i])
                cont_summr.append((sum, cont))
        dframe = pd.DataFrame(cont_summr)
        dframe.to_csv(dframe, encoding='utf-8', sep='\t')


    # write processed content summary (word_count) into file
    def process_content_sumamry(self):
        with open(CONTENT_PATH, 'rt', encoding='utf-8') as file_reader:
            lines = file_reader.read()
        text = ''
        with open(CONTENT_PATH_PROCESSED, 'wt', encoding='utf-8') as file_writer:
            for line in lines.splitlines():
                line = self.preprocess_paragraph(line)
                if len(line.split(' ')) > 4:
                    file_writer.write(line.strip() + '\n')
                text += line
        tokens = word_tokenize(text)
        word_count = self.token_statictics(tokens)
        self.write_wcount(word_count)

    def word_tokenize(self, sent):
        tokens = word_tokenize(sent)
        tkn_list = list()
        for i, w in enumerate(tokens):
            dot_token = re.findall('(\w+)(\.)(\w+)', w)
            if dot_token:
                tokens[i] = dot_token
        return tokens

pre = DataPreprocess()
pre.write_csv_file()
