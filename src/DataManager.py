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
    def get_xtrain(self):
        pass

    


class DataManager(DataManagerInterFace):
    def __init__(self,reduction=None) -> None:
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

        


    def get_xtrain(self):
        # return (X_train,y_train)
        return self.xtrain
    
    def get_ytrain(self):
        # return (X_train,y_train)
        return self.ytrain

    def get_xvalid(self):
        # return (X_test,y_test)
        return self.xvalid
    
    def get_yvalid(self):
        return self.yvalid

    def get_max_len_sequence(self):
        return self.max_len_sequence
