# -*- coding: utf-8 -*- 

import os
import re
import numpy as np
from keras.preprocessing import sequence, text
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils, generic_utils
from keras.models import Sequential
from keras.layers.embeddings import WordContextProduct, Embedding


def text_generator(path=data_path, sentence_num=sentence_num):
	for line in open(data_path).readlines()[:sentence_num]:
		yield ' '.join(list(re.sub(ur'[^0-9a-zA-Z\u4e00-\u9fa5]', '', line.strip().decode('utf-8')))).encode('utf-8')

def closest_to_point(point, nb_closest=10):
    proximities = np.dot(norm_weights, point)
    tups = list(zip(list(range(len(proximities))), proximities))
    tups.sort(key=lambda x: x[1], reverse=True)
    return [(reverse_word_index.get(t[0]), t[1]) for t in tups[:nb_closest]]  

def closest_to_word(w, nb_closest=10):
    i = word_index.get(w)
    if (not i) or (i<skip_top) or (i>=max_features):
        return []
    return closest_to_point(norm_weights[i].T, nb_closest)


sentence_num = 20000
data_path = os.path.join('.','dp_fastfood_text.txt')

nb_epoch = 5
skip_top = 10
dim_proj = 256
max_features = 1000

tokenizer = text.Tokenizer(nb_words=max_features)
tokenizer.fit_on_texts(text_generator())

# ----- 训练 -----

model = Sequential()
model.add(WordContextProduct(max_features, proj_dim=dim_proj, init="uniform"))
model.compile(loss='mse', optimizer='rmsprop')

sampling_table = sequence.make_sampling_table(max_features)

for e in range(nb_epoch):
    print 'Epoch:', e
    progbar = generic_utils.Progbar(tokenizer.document_count)
    samples_seen, losses = 0, []
    for i, seq in enumerate(tokenizer.texts_to_sequences_generator(text_generator())):
        couples, labels = sequence.skipgrams(seq, max_features, window_size=4, negative_samples=1., sampling_table=sampling_table)
        if couples:
            X = np.array(couples, dtype="int32")
            loss = model.train_on_batch(X, labels)
            losses.append(loss)
            if len(losses) % 100 == 0:
                progbar.update(i, values=[("loss", np.mean(losses))])
                losses = []
            samples_seen += len(labels)

weights = model.layers[0].get_weights()[0]
weights[:skip_top] = np.zeros((skip_top, dim_proj))
norm_weights = np_utils.normalize(weights)
del model

word_index = tokenizer.word_index
reverse_word_index = dict([(v, k) for k, v in list(word_index.items())])

# ----- 测试 -----

words = ["我"]

for w in words:
    print '='*4, w, '='*4
    for r in closest_to_word(w):
        print r[0], r[1]
