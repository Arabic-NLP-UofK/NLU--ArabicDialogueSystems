from utils import sigmoid
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
np.random.seed(1)


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map["سلام"].shape[0]
    emb_matrix = np.zeros((vocab_len,emb_dim))
                                            
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
        
    embedding_layer = Embedding(vocab_len, emb_dim, trainable = False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer

def deepLSTM(input_shape, word_to_vec_map,word_to_index,dropout_prob,n_layers,n_h):
    sentence_indices = Input(shape=input_shape)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map,word_to_index)
    embeddings = embedding_layer(sentence_indices)
    X=embeddings
    for i in range(n_layers-1):
        X = LSTM(n_h ,return_sequences=True)(X)
        X = Dropout(dropout_prob)(X)
    
    X = LSTM(128,return_sequences=False)(X)
    X = Dropout(dropout_prob)(X)
    X = Dense(5)(X)
    X = Activation('sigmoid')(X)
    model = Model(inputs=sentence_indices, outputs=X)
    return model