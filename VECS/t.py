
# import fasttext
# import fasttext.util

# ft = fasttext.load_model('cc.en.300.bin')
# print(ft.get_dimension())

# fasttext.util.reduce_model(ft, 100)
# print(ft.get_dimension())


# import fasttext
# import os
# import numpy as np
# pwd = os.getcwd()
# model_bin = "/home/bringtree/data/wiki.zh.bin"
# model_vec = "/home/bringtree/data/wiki.zh.vec"
# model_bin = "model.bin"
# model = fasttext.load_model(model_bin)
# word_1 = model.get_word_vector('asdhasjhdkajshd')
# print(word_1[:20])


# from gensim.models import Word2Vec, KeyedVectors

# model = KeyedVectors.load_word2vec_format("wiki.hi.align.vec")


# model.wv.save_word2vec_format('model.bin', binary=True)




from gensim.models.wrappers import FastText

model = FastText.load_fasttext_format('model.bin')

print(model.most_similar('teacher'))
# Output = [('headteacher', 0.8075869083404541), ('schoolteacher', 0.7955552339553833), ('teachers', 0.733420729637146), ('teaches', 0.6839243173599243), ('meacher', 0.6825737357139587), ('teach', 0.6285147070884705), ('taught', 0.6244685649871826), ('teaching', 0.6199781894683838), ('schoolmaster', 0.6037642955780029), ('lessons', 0.5812176465988159)]

print(model.similarity('teacher', 'teaches'))
# Output = 0.683924396754