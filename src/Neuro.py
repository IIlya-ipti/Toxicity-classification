from abc import ABC, abstractmethod
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU,SimpleRNN,Embedding,BatchNormalization
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import BatchNormalization
from keras import optimizers
import tensorflow as tf



class NeuroParent(ABC):
    
    @classmethod
    @abstractmethod
    def train(self):
        pass

    @classmethod
    @abstractmethod
    def test(self):
        pass

class FirstNeuro(NeuroParent):
    def __init__(self, dm):
        self.data_manager = dm
        self.model = Sequential()
        self.model.add(Embedding(len(dm.get_word_index()) + 1,
                40,
                input_length=dm.get_max_len_sequence()))
        self.model.add(GRU(10))
        self.model.add(Dense(1,activation='tanh'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    def train(self):

        self.model.fit(*self.data_manager.get_train_data(), epochs=5, batch_size=20)

    def test(self):
        return self.model.predict(self.data_manager.get_test_data()[0])

class SecondNeuto(NeuroParent):
    def __init__(self, word_index, embedding_matrix, max_len=1500) :
        self.model = Sequential()
        self.model.add(Embedding(len(word_index) + 1,
                    300,
                    weights=[embedding_matrix],
                    input_length=max_len,
                    trainable=False))

        self.model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    def train(self, xtrain, ytrain):
        self.model.fit(xtrain, ytrain, epochs=5, batch_size=64)

    def test(self, xvalid):
        return self.model.predict(xvalid)

'''class ThirdNeuto(NeuroParent):
    def __init__(self, word_index, embedding_matrix,max_len=1500):
        model = Sequential()
        model.add(Embedding(len(word_index) + 1,
                    300,
                    weights=[embedding_matrix],
                    input_length=max_len,
                    trainable=False))
        model.add(SpatialDropout1D(0.3))
        model.add(GRU(300))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])   
    def train(self, xtrain, ytrain):

        self.model.fit(xtrain, ytrain, nb_epoch=5, batch_size=64)

    def test(self, xvalid):
        return self.model.predict(xvalid)'''

class testNeuro:
    def __init__(self) -> None:
        pass

    def train(self):
        pass
    
    def test(self):
        pass

