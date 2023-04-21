
from keras.preprocessing.text import Tokenizer
import pickle
from keras.utils import pad_sequences
from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
from tokenizers import BertWordPieceTokenizer
import transformers

class ITokenWrapper(ABC):
    
    @classmethod
    @abstractmethod
    def get_features(self):
        pass

    @classmethod
    @abstractmethod
    def load(self):
        pass
    
    


    

class TokenSimple(ITokenWrapper):
    def __init__(self, xtrain, xvalid, max_len_sequence, load_mode=False):
        self.max_len_sequence = max_len_sequence
        self.xtrain = xtrain
        self.xvalid = xvalid
        if load_mode:
            self.load()
        else:
            self.save()
    def __init__(self, max_len_sequence=1503):
        self.max_len_sequence = max_len_sequence
        self.load()

    def load(self):
        with open('data/tokenizer.pickle', 'rb') as handle:
            self.token = pickle.load(handle)
    
    def save(self):
        self.token = Tokenizer(num_words=None)
        self.token.fit_on_texts(list(self.xtrain) + list(self.xvalid))  
        with open('data/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.token, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_features(self, data):
        return pad_sequences(self.token.texts_to_sequences(data),maxlen=self.max_len_sequence)

    def get_word_index(self):
        return self.token.word_index
    
    def get_max_len(self):
        return self.max_len_sequence