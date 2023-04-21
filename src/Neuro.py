from abc import ABC, abstractmethod
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU,SimpleRNN,Embedding,BatchNormalization
from tensorflow.keras import layers 
from keras.layers.core import Activation, Dropout
from tensorflow.keras.layers import Dense, Input
import transformers
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from .Transformer import *

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
class NeuroParent(ABC):
    
    @classmethod
    @abstractmethod
    def train(self):
        pass

    @classmethod
    @abstractmethod
    def test(self):
        pass

    def train(self ,x_train,y_train, x_val,y_val):
        self.model.fit(self.token.get_features(x_train),y_train,
                    validation_data=(self.token.get_features(x_val), y_val), epochs=2, batch_size=10)

    def test(self, xvalid, yvalid):
        scores = self.model.predict(self.token.get_features(xvalid))
        print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))
    
    def predict(self,xdata):
        return self.model.predict(self.token.get_features(xdata))
    
    def save_model(self, path):
        self.model.save_weights(path)
    
    def load_model(self,path):
        self.model.load_weights(path)
    
def roc_auc(predictions,target):
    '''
    This methods returns the AUC Score when given the Predictions
    and Labels
    '''
    
    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

class LSTMNeuro(NeuroParent):
    def __init__(self, tok):
        self.token = tok
        self.model = Sequential()
        self.model.add(Embedding(len(tok.get_word_index()) + 1,
                300,
                input_length=tok.get_max_len()))
        self.model.add(LSTM(100,dropout=0.3))
        self.model.add(Dense(1, activation='tanh'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


class TransformerNeuro(NeuroParent):
    def __init__(self, tok):
        self.token = tok
        embed_dim = 300  # Embedding size for each token
        num_heads = 2  # Number of attention heads
        ff_dim = 32  # Hidden layer size in feed forward network inside transformer

        inputs = layers.Input(shape=(tok.get_max_len(),))
        embedding_layer = TokenAndPositionEmbedding(tok.get_max_len(), len(tok.get_word_index()) + 1, embed_dim)
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        x = transformer_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(20, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(2, activation="softmax")(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )