from anago.trainer import Trainer
from anago.preprocessing import IndexTransformer
import gensim
import numpy as np
import anago
from anago.tagger import Tagger
from anago.utils import load_data_and_labels,filter_embeddings
from gensim.models.keyedvectors import KeyedVectors

if __name__=="__main__":
	wv_model = gensim.models.Word2Vec.load("wiki_cbow_100/wikipedia_cbow_100").wv
	train_path =  '../../data/collected/NER/train.txt'
	valid_path =  '../../data/collected/NER/valid.txt'

	print('Loading data...')
	x_train, y_train = load_data_and_labels(train_path)
	x_valid, y_valid = load_data_and_labels(valid_path)
	print("got ",len(x_train)," entries for training and ", len(x_valid), " entries for testing")
	entities=set()
	for s in y_train:
	    for w in s:
	        entities.add(w)
	print("Defined entities are :",entities)

	preprocessor = IndexTransformer(use_char=True)
	x = x_train+x_valid
	y = y_train+y_valid
	preprocessor.fit(x,y)
	print(len(x_train), 'train sequences')
	print(len(x_valid), 'valid sequences')



	embeddings = filter_embeddings(wv_model, preprocessor._word_vocab.vocab, wv_model.vector_size)
	# Use pre-trained word embeddings

	model = anago.models.BiLSTMCRF(embeddings=embeddings,
	                               use_crf=False,
	                               use_char=True,
	                               num_labels=preprocessor.label_size,
	                               word_vocab_size=preprocessor.word_vocab_size,
	                               char_vocab_size=preprocessor.char_vocab_size,
	                               dropout=.5,
	                               word_lstm_size=120
	                              )
	model.build()
	model.compile(loss=model.get_loss(), optimizer='adam',metrics=["acc"])
	model.summary()

	trainer = Trainer(model, preprocessor=preprocessor)
	trainer.train(x_train, y_train,x_valid=x_valid,y_valid=y_valid,epochs=100)