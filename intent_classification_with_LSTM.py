import numpy as np
import gensim
from keras.callbacks import EarlyStopping
from models import deepLSTM
from utils import read_csv,convert_to_bin
from utils import f1,precision,recall
from utils import int_to_intent,intent_to_int
from utils import sentences_to_indices

np.random.seed(1)

TRAIN_PATH = "../../data/collected/intent_classification/train.csv"
TEST_PATH = "../../data/collected/intent_classification/test.csv"
WORD_VECTORS_DIR = 'wiki_cbow_100/wikipedia_cbow_100'

if __name__=="__main__":
    # load the data into numpy arrays
    X_train, Y_train = read_csv(TRAIN_PATH)
    X_test, Y_test = read_csv(TEST_PATH)
    print("read ", X_train.shape[0]," entries for training")
    print("read ", X_test.shape[0]," entries for testing")

    #get the maximum lenght of the sentence 
    maxLen = 0
    for sentence in X_train:
        s_len = len(sentence.split())
        if s_len>maxLen:
            maxLen=s_len
    for sentence in X_test:
        s_len = len(sentence.split())
        if s_len>maxLen:
            maxLen=s_len
    print(maxLen)

    #load the word embeddings using gensim
    word_to_vec_map = gensim.models.Word2Vec.load(WORD_VECTORS_DIR)

    #build the vocabulary dictionary
    words = list(word_to_vec_map.wv.vocab.keys())
    word_to_index, index_to_word = dict(),dict()
    for i,word in enumerate(words):
        word_to_index[word]=i
        index_to_word[i]=word
    print("vocabulary size= ",len(words)," words")

    #create the model
    model = deepLSTM((maxLen,), word_to_vec_map.wv,word_to_index,.7,2,120)
    model.summary()

    model.compile(loss='binary_crossentropy'
                  , optimizer='adam'
                  , metrics=['accuracy',f1,precision,recall])

    X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
    Y_train_bin = convert_to_bin(Y_train, C = 5)


    X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
    Y_test_bin = convert_to_bin(Y_test, C = 5)

    # build the model using Early stopping to get the best set of parameters
    model.fit(X_train_indices, Y_train_bin
              , epochs = 100
              , validation_data=(X_test_indices,Y_test_bin)
               ,callbacks=[EarlyStopping(monitor='val_f1',mode='max',patience=10,restore_best_weights=True)]
              ,batch_size = 32
              , shuffle=True)

    # print the testing results
    loss, acc_,f1_,prec_,recall_ = model.evaluate(X_test_indices, Y_test_bin)
    print()
    print("Test accuracy = ", acc_)
    print("Test f1 = ", f1_)
    print("Test precision = ", prec_)
    print("Test recall = ", recall_)




    # This code allows you to see the mislabelled examples
    C = 5
    X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
    pred = model.predict(X_test_indices)
    pred = np.round(pred)
    for i in range(len(X_test)):
        x = X_test_indices
        pred_intents = pred[i]
        if((pred_intents == Y_test_bin[i]).sum()<pred_intents.shape[0]):
            print('Expected intent:'+ int_to_intent(Y_test[i]) + 
                  '\n\tprediction: '+ X_test[i] +'\n\t'+
                  int_to_intent(int("".join([str(int(_b)) for _b in pred_intents ]),2)).strip())


    #model.save(filepath="intent_classification_with_deep_lstm")