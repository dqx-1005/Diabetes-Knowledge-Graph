# encoding=utf8
import sys
import base64
from py2neo import Graph
import random
sys.path.append('../')
import re
import json
import pickle
import codecs
import numpy as np
import pickle as pkl
from Detect.model import Model
import tensorflow as tf
from six import iteritems
from Detect.utils import load_config
from dataProcess.config import Config
from Detect.utils import get_logger, create_model
from Detect.data_utils import load_word2vec, input_from_line
graph = Graph("http://localhost:7474", auth=("neo4j", "neo4j"),name="neo4j")  
rnn_layers = 2
embedding_size = 50
hidden_size = 50
input_dropout = 0.5
learning_rate = 0.001
max_grad_norm = 5
num_epochs = 1000
batch_size = 32
seq_length = 10
config_drug = Config()
restore_path = r'../WordEnhance/model'
all_whole_info = pkl.load(open('../processed/all_unique_entity.pkl', mode='rb'))


def load_vocab(vocab_file):
    with codecs.open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_index_dict = json.load(f)
    index_vocab_dict = {}
    vocab_size = 0
    for char, index in iteritems(vocab_index_dict):
        index_vocab_dict[index] = char
        vocab_size += 1
    return vocab_index_dict, index_vocab_dict, vocab_size


def cos(a, b):
    ma = np.linalg.norm(a)
    mb = np.linalg.norm(b)
    # cosine Similarity
    sim = (np.matmul(a, b)) / (ma * mb)
    return sim


def find_lcsubstr_spe(s1, s2_):
    max_length_ff = ''
    qwe = 0
    original = ''
    for s2 in s2_:
        m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
        mmax = 0
        p = 0
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    m[i+1][j+1] = m[i][j]+1
                    if m[i+1][j+1] > mmax:
                        mmax = m[i+1][j+1]
                        p = i+1
        if len(max_length_ff) < len(s1[p-mmax:p]):
            max_length_ff = s1[p-mmax:p]
            qwe = mmax
            original = s2
    # print(max_length_ff, qwe)
    return original, qwe


def find_lcsubstr(s1, s2_):
    max_length_ff = ''
    original = ''
    for s2 in s2_:
        m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
        mmax = 0
        p = 0
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    m[i+1][j+1] = m[i][j]+1
                    if m[i+1][j+1] > mmax:
                        mmax = m[i+1][j+1]
                        p = i+1
        if len(max_length_ff) < len(s1[p-mmax:p]):
            max_length_ff = s1[p-mmax:p]
            original = s2
    # print(max_length_ff, qwe)
    return original, max_length_ff


def find_lcsubstr_signal(s1, s2):
    max_length_ff = ''
    qwe = 0
    original = ''
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j]+1
                if m[i+1][j+1] > mmax:
                    mmax = m[i+1][j+1]
                    p = i+1
    if len(max_length_ff) < len(s1[p-mmax:p]):
        max_length_ff = s1[p-mmax:p]
        qwe = mmax
        original = s2
    # print(max_length_ff, qwe)
    return original, qwe


def de_digit(txt_cur):
    # print(txt_cur)
    mo_name_char = list(txt_cur)
    ff = re.findall(r'[a-z]|[0-9]', txt_cur)
    for fff in ff:
        mo_name_char.remove(fff)
    return ''.join(mo_name_char)


def evaluate_line():
    config = load_config('config_file')
    logger = get_logger('log/train.log')
    entity_saved = open('../test_data/entity_test_result_predict.txt', mode='w', encoding='utf-8')
    entity_dict = []
    #
    with open("maps.pkl", "rb") as f:  # load graph config
        # print(pickle.load(f))
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
        # print(char_to_id)
        # print(id_to_char)
        # print(tag_to_id)
        # print(id_to_tag)
    with trainer_one_graph.as_default():
        model = create_model(trainer_one_session, Model, 'ckpt/', load_word2vec, config, id_to_char, logger, False)
        # relation_dict = pkl.load(open('all_possible_subject_object_type_2_name.pkl', mode='rb'))
        # relation_dict1 = pkl.load(open('../DataProcess/Source/all_relation_id_name.pkl', mode='rb'))
        # cur_all_entity_pairs = pkl.load(open('../DataProcess/Source/cur_all_entity_pairs.pkl', mode='rb'))
        # print('relation_dict:', relation_dict)
        # print('relation_dict1:', relation_dict1)
        global_unique = []
        with trainer_one_session.as_default():
            test_data = open('../test_data/test_99.txt', mode='r', encoding='utf-8').readlines()
            for cur_txt in test_data:
                cur_txt = cur_txt.replace(' ', '')
                entity_predicted_value, tags = model.evaluate_line(trainer_one_session, input_from_line(cur_txt.strip(), char_to_id), id_to_tag)
                # check result
                if entity_predicted_value is not None:
                    cur_entity = entity_predicted_value['entities']
                    print('cur_entity:', cur_entity)
                    entity_saved.write(str(cur_entity) + '\n')
                    entity_dict.append(cur_entity)
        pkl.dump(entity_dict, open('../test_data/entity_dict', mode='wb'))


def one_step(trainer_one_graph, trainer_one_session, txt='werg'):
    config ={
        "model_type": "gru",
        "num_chars": 2007,
        "char_dim": 100,
        "num_tags": 37,
        "seg_dim": 20,
        "lstm_dim": 100,
        "batch_size": 32.0,
        "emb_file": "Source/word_vec.txt",
        "clip": 5.0,
        "dropout_keep": 0.5,
        "optimizer": "adam",
        "lr": 0.001,
        "tag_schema": "iob",
        "pre_emb": True,
        "zeros": True,
        "lower": True
    }
    logger = get_logger('../Detect/log/train.log')
    #
    with open("../Detect/maps.pkl", "rb") as f:  # load graph config
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
        # print('char_to_id, id_to_char, tag_to_id, id_to_tag:', len(char_to_id), len(id_to_char), len(tag_to_id), len(id_to_tag))
    with trainer_one_graph.as_default():
        model = create_model(trainer_one_session, Model, '../Detect/ckpt/', load_word2vec, config, id_to_char, logger, False)
        with trainer_one_session.as_default():
            txt = txt.replace(' ', '')
            entity_predicted_value, tags = model.evaluate_line(trainer_one_session,
                                                               input_from_line(txt.strip(), char_to_id),
                                                               id_to_tag)
            # check result
            if entity_predicted_value is not None:
                cur_entity = entity_predicted_value['entities']
                # print('cur_entity:', cur_entity)
                try:
                    print('cur_entity:', cur_entity[0])
                    cur_entity = cur_entity[0]['word']
                    return cur_entity
                except Exception as ex:
                    pass
            return None


# mapping
def main(original, start_text):
    max_watch = ''
    max_keshi = ''
    best_score = 0
    res_all = {}
    # print('test:::::::::::::::::::::::::::::::::::::::', original, start_text)

    for j in all_whole_info:
        if start_text is not None:
            final_matched, max_len = find_lcsubstr_signal(start_text, j)  # LCS 比较
        else:
            final_matched, max_len = find_lcsubstr_signal(original, j)  # LCS 比较
        if start_text is not None:
            if max_len >= len(start_text) * 2 / 3:
                res_all[final_matched] = max_len
        else:
            if max_len >= len(original) * 2 / 3:
                res_all[final_matched] = max_len
    res_all = sorted(res_all.items(), key=lambda item: item[1], reverse=True)
    # print('res_all:', res_all)
    # MATCH p=({entity:"{T48}"})-[r:ADE_Drug]->() RETURN p LIMIT 25
    # data = graph.run("match(p:{name:'{}'}) -[r:{}]->(n) return p.name, r, n.name limit 50".format(res_all[0][0], config_drug.config["2"]))
    # data = graph.run("match(p:{name:'{}'}) -[r:{}]->(n) return p limit 50".format(res_all[0][0], config_drug.config["2"]))
    data = []
    for top_2_name in res_all[: 2]:
        data_top = []
        for xxx_dy in ['疾病症状', '症状', '表现', '临床表现']:
            if xxx_dy in original:
                data_top = list(graph.run("MATCH p=({name:'%s'})-[r:%s]->() RETURN p LIMIT 25" % (format(top_2_name[0]), config_drug.config["2"])))
                print('search:', "MATCH p=({name:'%s'})-[r:%s]->() RETURN p.name" % (format(top_2_name[0]), config_drug.config["2"]))
        for xxx_dy in ['疾病病因', '病原', '病因']:
            if xxx_dy in original:
                data_top = list(graph.run("MATCH p=({name:'%s'})-[r:%s]->() RETURN p LIMIT 25" % (format(top_2_name[0]), config_drug.config["3"])))
                print('search:', "MATCH p=({name:'%s'})-[r:%s]->() RETURN p.name" % (format(top_2_name[0]), config_drug.config["3"]))
        for xxx_dy in ['忌用', '忌用药', '副作用', '副作用药']:
            if xxx_dy in original:
                data_top = list(graph.run("MATCH p=({name:'%s'})-[r:%s]->() RETURN p LIMIT 25" % (format(top_2_name[0]), config_drug.config["0"])))
                print('search:', "MATCH p=({name:'%s'})-[r:%s]->() RETURN p.name" % (format(top_2_name[0]), config_drug.config["0"]))
        for xxx_dy in ['那种病', '哪种病', '疾病类别', '属于那种', '属于哪种']:
            if xxx_dy in original:
                data_top = list(graph.run("MATCH p=({name:'%s'})-[r:%s]->() RETURN p LIMIT 25" % (format(top_2_name[0]), config_drug.config["1"])))
                print('search:', "MATCH p=({name:'%s'})-[r:%s]->() RETURN p.name" % (format(top_2_name[0]), config_drug.config["1"]))
        for xxx_dy in ['怎么和', '怎么喝', '怎么服用', '一次喝多少', '怎么用']:
            if xxx_dy in original:
                data_top = list(graph.run("MATCH p=({name:'%s'})-[r:%s]->() RETURN p LIMIT 25" % (format(top_2_name[0]), config_drug.config["6"])))
                print('search:', "MATCH p=({name:'%s'})-[r:%s]->() RETURN p.name" % (format(top_2_name[0]), config_drug.config["6"]))
        for xxx_dy in ['服用', '喝什么药', '吃什么药', '服用什么药', '买什么药']:
            if xxx_dy in original:
                data_top = list(graph.run("MATCH p=({name:'%s'})-[r:%s]->() RETURN p LIMIT 25" % (format(top_2_name[0]), config_drug.config["5"])))
                print('search:', "MATCH p=({name:'%s'})-[r:%s]->() RETURN p.name" % (format(top_2_name[0]), config_drug.config["5"]))

        if len(data_top) > 0:
            data.append([top_2_name[0], data_top])
    print('------问题：%s' % original + '------')
    print('推荐结果:')
    for cur_res in data:
        qwer = []
        j_name = cur_res[0]
        for j in cur_res[1]:
            j_res = list(j)
            # print('j_res:', j_res)
            # qwer.append(str(j_res).split('->')[-1].replace('(', '').replace(')]', ''))
            qwer.append(str(j_res).split(', name=')[-1].replace('(', '').replace(')', '').replace(']', '').replace("'", ''))
        qwer = j_name + ":" + ' 、'.join(qwer)
        print('\t' + qwer)
    # data = graph.run("MATCH p=({name:'%s'})-[]->() RETURN p LIMIT 25" % (res_all[1][0]))
    return {"data": res_all}
    # return {"data": 'None'}


if __name__ == "__main__":
    vm = ''
    test = ''
    while not vm.__eq__('@'):
        vm = input().strip()
        # test = input().strip()
        # if test == '\\n':
        trainer_one_graph = tf.Graph()  # entity detect
        trainer_one_session = tf.Session(graph=trainer_one_graph)
        g = tf.Graph()  # entity detect
        session = tf.Session(graph=g)
        entity_detect = one_step(trainer_one_graph, trainer_one_session, vm)
        # print('entity_detect；', entity_detect)
        # if entity_detect is not None:
        main(vm, entity_detect)



