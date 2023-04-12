# from gensim.models import Word2Vec
# from gensim.corpora.dictionary import Dictionary
# import pickle
#
#
# def create_dictionaries(model):
#     gensim_dict = Dictionary()
#     gensim_dict.doc2bow(model.wv.key_to_index.keys(), allow_update=True)
#
#     w2idx = {v: k + 1 for k, v in gensim_dict.items()}
#     w2vec = {word: model.wv[word] for word in w2idx.keys()}
#     return w2idx, w2vec
#
#
# model = Word2Vec.load('./dataset/w2v_dim3.model')
# index_dic, word_vectors = create_dictionaries(model)
#
# output = open('./dataset/vocab_dim3.pkl', 'wb')
# pickle.dump(index_dic, output)
# pickle.dump(word_vectors, output)
# output.close()
import numpy as np
import pandas as pd
path1 = "./dataset/test.csv"
data = pd.read_csv(path1, encoding='utf-8')
text = data['text']
sents = text.tolist()
new_sentences = []
for sent in sents:
    new_sen = []
    words = sent.split(' ')
    for word in words:
        try:
            new_sen.append(word)
        except:
            new_sen.append(0)

    new_sentences.append(new_sen)
new_sentences = np.array(new_sentences)
print(type(new_sentences))
print(type(new_sentences[0]))