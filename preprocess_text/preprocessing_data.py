import math
import operator
import nltk
import pandas as pd
from nltk import word_tokenize

from config import *
import re

class DataPreprocess:
    def __init__(self):
        self.resources_path = RESOURCES_PATH
        self.content, self.summary = self.read_file(4, 5)
        self.tokens = self.read_tokens()

    def read_file(self, sum_ind, cont_ind):
        d_frame = pd.read_csv(self.resources_path, skiprows=1, encoding='latin-1')
        content = list(d_frame.iloc[:, cont_ind])
        summary = list(d_frame.iloc[:, sum_ind])
        content_processed = list()
        summary_processed = list()
        data_len = len(content)
        for i in range(0, data_len):
            if (isinstance(content[i], str) and isinstance(summary[i], str)):
                content_processed.append(content[i])
                summary_processed.append(summary[i])
        tokens = ' '.join(list(content_processed))
        tokens = nltk.word_tokenize(tokens)
        return content, summary

    def read_tokens(self):
        d_frame = pd.read_csv(RESOURCES_PROCESSED, encoding='latin-1')
        content = list(d_frame.iloc[:, 1])
        tokens = ' '.join(content)
        tokens = nltk.word_tokenize(tokens)
        return tokens

    # return dictionary word_count {word:count}
    def token_statictics(self):
        word_count = dict()
        for token in self.tokens:
            if token not in word_count:
                word_count[token] = 1
            else:
                word_count[token] += 1
        return word_count

    # write word_count into file
    def write_wcount(self):
        word_count = self.token_statictics()
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
        para = str(para)
        para = re.sub('Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday', 'DAYOFWEEK', para)
        para = re.sub('\(HT\sPhoto\)|\(HT\?FILE\)|\(HT\?File\)|\(HT File\)|\(Picture: Sourced\)|\(HTPhoto\)', '', para)
        para = re.sub('(ALSO READ.*)|(Also Read.*)|(also Read.*)|(Also read.*)', '', para)
        para = re.sub('\d+\.\d{2}\s?(pm|am)?', 'HOUR', para)
        para = re.sub('\d+:\d{2}(am|pm)', 'HOUR', para)
        para = re.sub('(19|20)\d{2}', 'YEAR', para)
        para = re.sub('January|February|March|April|May|June|Junly'
                      '|August|September|October|November|December', 'MONTH', para)
        para = re.sub('#\w+', 'HASHTAG', para)
        para = re.sub('^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)'
                      '|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])'
                      '|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$',
                      'EMAIL', para)
        para = re.sub('(\d+)(-\d+){1,}', 'NUMBER', para)
        para = re.sub('\d+,\d+', 'NUMBER', para)
        para = re.sub('\d+,\d+', 'NUMBER', para)
        para = re.sub('^\d+\.\d+$', 'NUMBER', para)
        para = re.sub('((\d+\.\d+)|(\d+))\/\d+', 'NUMBER', para)
        para = re.sub('£\d+,\d+', 'MONEY', para)
        para = re.sub('^((http[s]?|ftp):\/)?\/?([^:\/\s]+)((\/\w+)*\/)'
                      '([\w\-\.]+[^#?\s]+)(.*)?(#[\w\-]+)?$', 'URL', para)
        para = re.sub('pic.twitter.com\/\w+|twitter.com/\w+', 'URL', para)
        para = re.sub('https?:[^\s]+', 'URL', para)
        para = re.sub('\w+\.twitter.com/\w+', 'URL', para)
        para = re.sub('www(\.\w+)+', 'URL', para)
        para = re.sub('\{"@context.*}}', '', para)
        para = re.sub("([a-zA-Z]+)?\*+([a-zA-Z])?", 'F-WORD', para)
        para = re.sub('fakenote_boxwrap.*differenceNUM\.', '', para)
        para = re.sub('\?|"|\.{2,}', " ", para)
        para = re.sub('\d+', 'NUMBER', para)
        para = re.sub('(@\w+)+', 'AT_TAG', para)
        para = re.sub('\(.+\)', ' ', para)
        para = re.sub(',', ' ', para)
        para = re.sub('-', ' ', para)
        para = re.sub('\.\w+', ' ', para)
        para = re.sub("'\w+'", ' ', para)
        para = re.sub("\.", ' ', para)
        para = re.sub("\/", ' ', para)
        para = re.sub("'", ' ', para)
        para = para.lower()
        para = self.adding_wspace(para)
        para = self.eliminate_non_character(para)
        return para

    def eliminate_non_character(self, para):
        tokens = word_tokenize(para)
        tkn = list()
        for token in tokens:
            try:
                temp = token.encode('utf-8').decode('ascii')
                tkn.append(temp)
            except UnicodeDecodeError:
                tkn.append('unknown_tag')
        return ' '.join(tkn)

    # adding white space to paragraph ago.Sources ==> ago. Sources
    def adding_wspace(self, para):
        fiall = re.findall('[a-zA-Z]+\.[a-zA-Z]+', para)
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
            if cont and isinstance(self.summary[i], str):
                cont = self.preprocess_paragraph(cont)
                sum = 'S.O.S ' + self.preprocess_paragraph(self.summary[i]) + ' E.O.S'
                if len(cont.split(' ')) > 2 and len(sum.split(' ')) > 2:
                    cont_summr.append((sum, cont))
        dframe = pd.DataFrame(cont_summr)
        dframe.to_csv(RESOURCES_PROCESSED, encoding='utf-8', header=False, index=False, index_label=False)

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

    def write_content_summar(self):
        d_frame = pd.read_csv(RESOURCES_PROCESSED, encoding='utf-8')
        summary = list(d_frame.iloc[:, 0])
        content = list(d_frame.iloc[:, 1])
        total = summary + content
        with open(CONTENT_PATH, 'wt', encoding='utf-8') as file_writer:
            for text in total:
                file_writer.write(text + '\n')


pre = DataPreprocess()
pre.write_content_summar()