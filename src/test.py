from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.preprocessing import sequence, text
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU,SimpleRNN,Embedding,BatchNormalization
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping


import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff




token = Tokenizer(num_words=None)

max_len = 5
word = ['lol of the lol lol the the of of']
token.fit_on_texts(word)

print(token.word_index)
print(token.texts_to_sequences(word))
# Asprint(pad_sequences(token.texts_to_sequences(word),maxlen=max_len))

word_index=token.word_index


model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     300,
                     input_length=max_len))
model.add(SimpleRNN(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


