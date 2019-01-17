import numpy as np
import csv
from normalizer import normalize
import keras.backend as K
np.random.seed(1)


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
    
def f1(y_true, y_pred):
    _precision = precision(y_true, y_pred)
    _recall = recall(y_true, y_pred)
    return 2*((_precision*_recall)/(_precision+_recall+K.epsilon()))

def read_csv(filename):
    phrase = []
    label = []
    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            y=row[1].strip('"')
            y_int=intent_to_int(y)
            if y_int>0:
                phrase.append(row[0].strip().strip('"'))
                label.append(y_int)
    X = np.asarray(phrase)
    Y = np.asarray(label, dtype=int)
    return X, Y

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

def int_to_intent(y_int):
    intents = ""
    greeting = y_int - ((y_int>>1)*2)
    if greeting:
        intents +="greeting, "
    y_int=y_int>>1
    
    inform = y_int - ((y_int>>1)*2)
    if inform:
        intents +="inform, "
    y_int=y_int>>1
    
    chat = y_int - ((y_int>>1)*2)
    if chat:
        intents += "chat, "
    y_int=y_int>>1
    
    check_status = y_int - ((y_int>>1)*2)
    if check_status:
        intents += "check_status, "
    y_int=y_int>>1
    
    goodbye = y_int - ((y_int>>1)*2)
    if goodbye:
        intents+="goodbye"
    y_int=y_int>>1
    assert y_int==0
    
    return intents
def convert_to_bin(Y,C):
    res = np.zeros((Y.shape[0],C))
    for i in range(Y.shape[0]):
        res[i,:]=[int(_b) for _b in format(Y[i], '0'+str(C)+'b')]
    return res


def sigmoid(z):
    return 1/(1+np.exp(-z))

def sentences_to_indices(X, word_to_index, max_len):
    
    m = X.shape[0]
    
    X_indices = np.zeros((m, max_len),dtype=int)
    
    for i in range(m):
        
        sentence_words = normalize(X[i]).split()
        
        j = 0
        for w in sentence_words:
            if w in word_to_index:
                X_indices[i, j] = word_to_index[w]
            j = j+1
                
    return X_indices