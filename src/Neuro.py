from abc import ABC, abstractmethod
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU,SimpleRNN,Embedding,BatchNormalization
from keras.layers.core import Activation, Dropout
from tensorflow.keras.layers import Dense, Input
import transformers
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

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
def roc_auc(predictions,target):
    '''
    This methods returns the AUC Score when given the Predictions
    and Labels
    '''
    
    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

class FirstNeuro(NeuroParent):
    def __init__(self, tok, max_len):
        self.token = tok
        self.model = Sequential()
        self.model.add(Embedding(len(tok.get_word_index()) + 1,
                300,
                input_length=max_len))
        self.model.add(LSTM(50,dropout=0.3))
        self.model.add(Dense(1))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    def train(self):

        self.model.fit(*self.token.get_features(), epochs=5, batch_size=20)

    def test(self):
        scores = self.model.predict(self.tok.get_xvalid)
        print("Auc: %.2f%%" % (roc_auc(scores,self.tok.get_yvalid())))

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


class Bert(NeuroParent):
    def __init__(self, max_len=512) -> None:
    
        input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        transformer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
        sequence_output = transformer(input_word_ids)[0]
        cls_token = sequence_output[:, 0, :]
        out = Dense(1, activation='sigmoid')(cls_token)
    
        self.model = Model(inputs=input_word_ids, outputs=out)
        self.model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, x_train, y_train, x_valid, y_valid):
        AUTO = tf.data.experimental.AUTOTUNE
        BATCH_SIZE = 128

        train_dataset = (
            tf.data.Dataset
            .from_tensor_slices((x_train, y_train))
            .repeat()
            .shuffle(2048)
            .batch(BATCH_SIZE)
            .prefetch(AUTO))
        valid_dataset = (
            tf.data.Dataset
            .from_tensor_slices((x_valid, y_valid))
            .batch(BATCH_SIZE)
            .cache()
            .prefetch(AUTO))

        self.model.fit(
            train_dataset,
            steps_per_epoch=x_train.shape[0],
            validation_data=valid_dataset,
            epochs=5)
        def test(self, test_dataset):
            return self.model.predict(test_dataset, verbose=1)
class testNeuro:
    def __init__(self) -> None:
        pass

    def train(self):
        pass
    
    def test(self):
        pass

