from DataManager import DataManager
from Neuro import *
from keras.layers.core import Dense, Activation, Dropout, Input

if __name__ == "__main__":
    a = [1,2,3,4]
    l = Input(shape=(2,))
    print(l)
    dm = DataManager(load=False)