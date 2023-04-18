import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences
import pickle
from abc import ABC, abstractmethod


train_path = r'data\jigsaw-toxic-comment-train.csv'
validation_path = r'data\test.csv'
test_path = r'data\validation.csv'

class DataManagerInterFace(ABC):
    
    @classmethod
    @abstractmethod
    def get_train_data(self):
        pass

    @classmethod
    @abstractmethod
    def get_test_data(self):
        pass

    @classmethod
    @abstractmethod
    def get_word_index(self):
        pass

    def __load_tokenizer(self):
        with open('data/tokenizer.pickle', 'rb') as handle:
            self.token = pickle.load(handle)

    def __save_tokenizer(self):
        with open('data/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.token, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def __to_sequence(self, data):
        return pad_sequences(self.token.texts_to_sequences(data),maxlen=self.max_len_sequence)

    def __init_tokenizer(self):
        self.token = Tokenizer(num_words=None)
        self.token.fit_on_texts(list(self.xtrain) + list(self.xvalid))

class DataManager(DataManagerInterFace):
    def __init__(self,load=False,reduction=None) -> None:
        train = pd.read_csv(train_path)
        validation = pd.read_csv(validation_path)
        test = pd.read_csv(test_path)
        
        # 
        train.drop(['severe_toxic','obscene','threat','insult','identity_hate'],axis=1,inplace=True)   

        if not(reduction is None):
            train = train.loc[:reduction,:]

        # init class data    
        self.max_len_sequence = train['comment_text'].apply(lambda x:len(str(x).split())).max() + 100
        self.xtrain, self.xvalid, self.ytrain, self.yvalid = train_test_split(train.comment_text.values, train.toxic.values, 
                                                  stratify=train.toxic.values, 
                                                  random_state=42, 
                                                  test_size=0.2, shuffle=True)

        if not load:
            self._DataManagerInterFace__init_tokenizer()
            self._DataManagerInterFace__save_tokenizer()
        else:
             self._DataManagerInterFace__load_tokenizer()

        self.xtrain_pad =  self._DataManagerInterFace__to_sequence(self.xtrain)
        self.xvalid_pad =  self._DataManagerInterFace__to_sequence(self.xvalid)
        self.word_index = self.token.word_index

    

    def get_train_data(self):
        # return (X_train,y_train)
        return (self.xtrain_pad,self.ytrain)

    def get_test_data(self):
        # return (X_test,y_test)
        return (self.xvalid_pad, self.yvalid)

    def get_word_index(self):
        return self.word_index
    
    def get_max_len_sequence(self):
        return self.max_len_sequence
