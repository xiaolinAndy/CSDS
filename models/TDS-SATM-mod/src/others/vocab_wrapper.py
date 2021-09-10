# -*- coding:utf-8 -*-
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
# from glove import Corpus
# from glove import Glove


class VocabWrapper(object):
    # Glove has not been implemented.
    def __init__(self, mode="word2vec", emb_size=100):
        self.mode = mode
        self.emb_size = emb_size
        self.model = None
        self.emb = None

    def _glove_init(self):
        pass

    def _word2vec_init(self):
        self.model = Word2Vec(size=self.emb_size, window=5, min_count=30, workers=4)

    def _glove_train(self, ex):
        pass

    def _word2vec_train(self, ex):
        if self.model.wv.vectors.size == 0:
            self.model.build_vocab(ex, update=False)
        else:
            self.model.build_vocab(ex, update=True)
        self.model.train(ex, total_examples=self.model.corpus_count, epochs=1)

    def _glove_report(self):
        pass

    def _word2vec_report(self):
        if self.model is not None:
            print("Total examples: %d" % self.model.corpus_count)
            print("Vocab Size: %d" % len(self.model.wv.vocab))
        else:
            print("Vocab Size: %d" % len(self.emb.vocab))

    def _glove_save_model(self, path):
        pass

    def _word2vec_save_model(self, path):
        self._word2vec_report()
        self.model.save(path)

    def _glove_load_model(self, path):
        pass

    def _word2vec_load_model(self, path):
        self.model = Word2Vec.load(path)

    def _glove_save_emb(self, path):
        pass

    def _word2vec_save_emb(self, path):
        self._word2vec_report()
        if self.model is not None:
            self.model.wv.save(path)
        else:
            self.emb.save(path)

    def _glove_load_emb(self, path):
        pass

    def _word2vec_load_emb(self, path):
        self.emb = KeyedVectors.load(path)
        self.emb_size = self.emb.vector_size

    def _w2i_glove(self, w):
        return None

    def _w2i_word2vec(self, w):
        if self.emb is not None:
            if w in self.emb.vocab.keys():
                return self.emb.vocab[w].index
        if self.model is not None:
            if w in self.model.wv.vocab.keys():
                return self.model.wv.vocab[w].index
        return None

    def _i2w_glove(self, idx):
        return None

    def _i2w_word2vec(self, idx):
        if self.emb is not None:
            if idx < len(self.emb.vocab):
                return self.emb.index2word[idx]
        if self.model is not None:
            if idx < len(self.model.wv.vocab):
                return self.model.wv.index2word[idx]
        return None

    def _i2e_glove(self, idx):
        return None

    def _i2e_word2vec(self, idx):
        if self.emb is not None:
            if idx < len(self.emb.vocab):
                return self.emb.vectors[idx]
        if self.model is not None:
            if idx < len(self.model.wv.vocab):
                return self.model.wv.vectors[idx]
        return None

    def _w2e_glove(self, w):
        return None

    def _w2e_word2vec(self, w):
        if self.emb is not None:
            if w in self.emb.vocab.keys():
                return self.emb[w]
        if self.model is not None:
            if w in self.model.wv.vocab.keys():
                return self.model.wv[w]
        return None

    def _voc_size_glove(self):
        return -1

    def _voc_size_word2vec(self):
        if self.emb is not None:
            return len(self.emb.vocab)
        if self.model is not None:
            return len(self.model.wv.vocab)
        return -1

    def _get_emb_glove(self):
        return None

    def _get_emb_word2vec(self):
        if self.emb is not None:
            return self.emb.vectors
        if self.model is not None:
            return self.model.wv.vectors
        return None

    def init_model(self):
        if self.mode == "glove":
            self._glove_init()
        else:
            self._word2vec_init()

    def train(self, ex):
        """
        ex: training examples.
            [['我', '爱', '中国', '。'],
             ['这', '是', '一个', '句子', '。']]
        """
        if self.mode == "glove":
            self._glove_train(ex)
        else:
            self._word2vec_train(ex)

    def report(self):
        if self.mode == "glove":
            self._glove_report()
        else:
            self._word2vec_report()

    def save_model(self, path):
        if self.mode == "glove":
            self._glove_save_model(path)
        else:
            self._word2vec_save_model(path)

    def load_model(self, path):
        if self.mode == "glove":
            self._glove_load_model(path)
        else:
            self._word2vec_load_model(path)

    def save_emb(self, path):
        if self.mode == "glove":
            self._glove_save_emb(path)
        else:
            self._word2vec_save_emb(path)

    def load_emb(self, path):
        if self.mode == "glove":
            self._glove_load_emb(path)
        else:
            self._word2vec_load_emb(path)

    def w2i(self, w):
        if self.mode == "glove":
            return self._w2i_glove(w)
        else:
            return self._w2i_word2vec(w)

    def i2w(self, idx):
        if self.mode == "glove":
            return self._i2w_glove(idx)
        else:
            return self._i2w_word2vec(idx)

    def w2e(self, w):
        if self.mode == "glove":
            return self._w2e_glove(w)
        else:
            return self._w2e_word2vec(w)

    def i2e(self, idx):
        if self.mode == "glove":
            return self._i2e_glove(idx)
        else:
            return self._i2e_word2vec(idx)

    def voc_size(self):
        if self.mode == "glove":
            return self._voc_size_glove()
        else:
            return self._voc_size_word2vec()

    def get_emb(self):
        if self.mode == "glove":
            return self._get_emb_glove()
        else:
            return self._get_emb_word2vec()
