# coding=utf-8
import os
import os
import re
import pickle as pkl
from gensim.models import word2vec
path = 'Testing'


def get_raw_content():
    path = '../../processed/labeled.txt'
    save_path = 'whole_summary_data.txt'
    data = open(path, mode='r', encoding='utf-8').readlines()
    save_data = open(save_path, mode='w', encoding='utf-8')
    for k in data:
        k = k.split('\t')
        assert len(k) == 2, "Error"
        k = ''.join(k[0])
        save_data.write(k + '\n')


def vec_train():
    # filename = 'train_source.txt'  # 所有的文本
    filename = 'whole_summary_data.txt'  # 所有的文本
    fin = open(filename, mode='r', encoding='utf-8').readlines()
    fin_save = open("train_r_vec.txt", mode='w', encoding='utf-8')  # 按字符分割的数据集
    for cur_line in fin:
        cur_line = ' '.join(list(cur_line)).strip()
        fin_save.write(cur_line + '\n')

    print('model Train.')
    sentences = word2vec.Text8Corpus("train_r_vec.txt")
    model = word2vec.Word2Vec(sentences)
    # model = word2vec.Word2Vec(ff, size=12, window=5, min_count=1, workers=5, sg=1, hs=1)
    # print('model save.')
    # # 4保存模型，以便重用
    model.save("test_01.model.vec")
    model.wv.save_word2vec_format('word_vec.txt', ' vec.vocab.txt', binary=False)
    # # 将模型保存成文本，model.wv.save_word2vec_format()来进行模型的保存的话，会生成一个模型文件。里边存放着模型中所有词的词向量。这个文件中有多少行模型中就有多少个词向量。


if __name__ == "__main__":
    get_raw_content()
    vec_train()
