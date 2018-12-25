import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import gensim
import re
from normalizer import normalize
import string
from gensim.models import FastText
from keras.callbacks import TensorBoard
from keras.callbacks import Callback, CallbackList
from keras import backend as K
from livelossplot import PlotLossesKeras
import matplotlib as plt
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from matplotlib import figure
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from nltk.corpus import stopwords
import nltk
from functools import reduce


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


MAX_NB_WORDS = 1500
MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 100

EMBEDDING_FILE = "tweets.bin"
#category_index = {"chat":0,"inform":1,"greeting":2, "goodbye":3, "check_status":4}
#category_reverse_index = dict((y,x) for (x,y) in category_index.items())
STOPWORDS = set(stopwords.words("arabic"))
def get_categories(number):
    intent = ""
    if number%2==1:
        intent+="greeting, "
    number = number//2
    if number%2==1:
        intent+="inform, "
    number = number//2
    if number%2==1:
        intent+="chat, "
    number = number//2
    if number%2==1:
        intent+="check_status, "
    number = number//2
    if number%2==1:
        intent+="goodbye, "
    number = number//2
    return intent
def convert_to_bin(Y,C):
    res = np.zeros((Y.shape[0],C))
    for i in range(Y.shape[0]):
        res[i,:]=[int(_b) for _b in format(Y[i], '0'+str(C)+'b')]
    return res
data = pd.read_csv("train.tsv", sep="\t")
def intent_to_int(y):
    y_int=0
    if "greeting" in y:
        y_int += 1<<0
    if "inform" in y:
        y_int += 1<<1
    if "chat" in y:
        y_int += 1<<2
    if "check_status" in y:
        y_int += 1<<3
    if "goodbye" in y:
        y_int += 1<<4
    return y_int
data["intent"]=data["intent"].apply(intent_to_int)

all_texts =data['command']
#all_texts = all_texts.drop_duplicates(keep=False)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(all_texts)

train_sequences = tokenizer.texts_to_sequences(data['command'])
train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
#test_sequences = tokenizer.texts_to_sequences(test['command'])
#est_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

word_index = tokenizer.word_index
test_sequence = tokenizer.texts_to_sequences(["باب مكيف مساء", "ضوء بيت انارة"])
padded_sequence = pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH)



category = data['intent'].values
print(category[0])
category = convert_to_bin(category,5)

VALIDATION_SPLIT = 0.3
indices = np.arange(train_data.shape[0]) # get sequence of row index
np.random.shuffle(indices) # shuffle the row indexes
data = train_data[indices] # shuffle data/product-titles/x-axis
category = category[indices] # shuffle labels/category/y-axis
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = category[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = category[-nb_validation_samples:]

model = gensim.models.Word2Vec.load("tweets_cbow_300")
vocab = model.wv.vocab.keys()
from keras.layers import Embedding
word_index = tokenizer.word_index
print(len(word_index))
nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, 300))
for word, i in word_index.items():
    if word in model.wv:
        embedding_matrix[i] = model.wv[ word ]


embedding_layer = Embedding(embedding_matrix.shape[0], # or len(word_index) + 1
                            embedding_matrix.shape[1], # or EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

model_1 = Sequential()
model_1.add(embedding_layer)
model_1.add(Conv1D(100,15,padding='valid',activation='relu',strides=1))
model_1.add(GlobalMaxPooling1D())
model_1.add(Dense(200))
model_1.add(Dropout(0.65))
model_1.add(Activation('relu'))
model_1.add(Dense(5))
model_1.add(Activation('sigmoid'))
model_1.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['acc',f1,precision,recall])
model_1.summary()

model_1.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=128, callbacks=[PlotLossesKeras()])
#history = model_1.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=128, callbacks=[PlotLossesKeras()])
score = model_1.evaluate(x_val, y_val, verbose=0)
print('Test loss:', score[0])
#print('Test accuracy:', round(score[1]*100,2), '&', round(score[2]*100,2), '&',
  #    round(score[3]*100,2), '&',round(score[4]*100,2) )
print('Test accuracy:', round(score[1]*100,2))
print('Test f1:', round(score[2]*100,2))
print('Test precision:', round(score[3]*100,2))
print('Test recall:', round(score[4]*100,2))

example_product = "صباح الخير , شغل المروحه"
example_sequence = tokenizer.texts_to_sequences([example_product])
example_padded_sequence = pad_sequences(example_sequence, maxlen=MAX_SEQUENCE_LENGTH)
_class = model_1.predict(example_padded_sequence, verbose=0)
_class=np.round(_class)
intent = 0
print(_class)
for i in range(_class.shape[1]):
    if _class[0,i]==1:
        intent+=2**(4-i)
print(intent)
print("-"*10)
print("Predicted category: ", get_categories(intent) )
print("-"*10)
