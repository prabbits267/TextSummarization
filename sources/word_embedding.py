from boto.s3.key import Key
from gensim.models import Word2Vec

from config import *

class WordEmbedding:
    def __init__(self):
        self.window_size = 5
        self.embed_size = 300
        self.min_count = 1
        self.sentences, self.token = self.read_text()
        self.word2vec = self.load_model()

    def read_text(self):
        with open(CONTENT_PATH, 'rt', encoding='utf-8') as file_reader:
            lines = file_reader.read()
            sentences = [w.split() for w in lines.splitlines()]
        return sentences, sorted(set(lines.split()))

    def load_model(self):
        try:
            model = Word2Vec.load(EMBEDDING_MODEL)
        except FileNotFoundError:
            print('Model not found. Training the model...')
            model = Word2Vec(self.sentences, window=self.window_size, size=self.embed_size, min_count=self.min_count)
            model.save(EMBEDDING_MODEL)
        return model

    def get_vector(self, word):
        try:
            return self.word2vec[word]
        except KeyError:
            print(word)
            return self.word2vec['unknown_tag']