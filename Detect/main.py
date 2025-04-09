# coding=utf-8
import tensorflow as tf
import numpy as np
from model import Model
import pickle as pkl
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_ner
from data_utils import load_word2vec, input_from_line, BatchManager
import os
import pickle
import itertools
from collections import OrderedDict

flags = tf.app.flags
flags.DEFINE_boolean("clean",       False,      "clean train folder")
flags.DEFINE_boolean("train",       True,      "Whether train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("gru_dim",    100,        "Num of hidden units in LSTM, or num of filters in IDCNN")
# flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")
flags.DEFINE_string("tag_schema",   "iob",    "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_float("batch_size",    32,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       True,       "Wither lower case")

flags.DEFINE_integer("max_epoch",   1000,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path", "result", "Path for results")
# flags.DEFINE_string("emb_file",     os.path.join("data", "word_vec.txt"),  "Path for pre_trained embedding")
flags.DEFINE_string("emb_file",     os.path.join("Source", "word_vec.txt"),  "Path for pre_trained embedding")
# flags.DEFINE_string("train_file",   os.path.join("../../Source/processed", "train_en.txt"),  "Path for train data")
flags.DEFINE_string("train_file",   os.path.join("../processed", "train_per_sentence.txt"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join("../processed", "train_per_sentence.txt"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join("../processed", "train_per_sentence.txt"),   "Path for test data")
# flags.DEFINE_string("model_type", "bilstm", "Model type, can be idcnn or bilstm")
flags.DEFINE_string("model_type", "gru", "Model type, can be idcnn or bilstm")

FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# config for the model
def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["model_type"] = 'gru'
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = 100
    config["batch_size"] = FLAGS.batch_size
    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config


def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    logger.info("Evaluation result: {}".format(eval_lines))
    if len(eval_lines) > 1:
        f1 = float(eval_lines[1].strip().split()[-1])
    else:
        f1 = 0
        logger.error("Insufficient evaluation output.")

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train():
    # load data sets
    # 训练集，测试集，验证集
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)
    # print('dev_sentences:', dev_sentences)
    # 采用不同序列标注法
    # Use selected tagging scheme (IOB / IOBES)
    update_tag_scheme(train_sentences, FLAGS.tag_schema)
    update_tag_scheme(test_sentences, FLAGS.tag_schema)
    update_tag_scheme(dev_sentences, FLAGS.tag_schema)
    # create maps if not exist
    if not os.path.isfile(FLAGS.map_file):
        # create dictionary for word
        # 预训练的嵌入模型
        if FLAGS.pre_emb:
            dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])
                )
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

        # Create a dictionary and a mapping for tags
        # tag_to_id是指序列标注法标注的tag对应的id
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        print("_t, tag_to_id, id_to_tag"", _t, tag_to_id, id_to_tag")
        # print(char_to_id)
        # print(id_to_char)
        # print(tag_to_id)
        # print(id_to_tag)
        # print('*' * 100)

        with open(FLAGS.map_file, "wb") as f:
            print('maps save:', len(char_to_id), len(id_to_char), len(tag_to_id), len(id_to_tag))
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
            # print('char_to_id, id_to_char, tag_to_id, id_to_tag:', char_to_id, id_to_char, tag_to_id, id_to_tag)

    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))

    train_manager = BatchManager(train_data, FLAGS.batch_size)
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)
    # make path for store log and model if not exist
    make_path(FLAGS)
    # if os.path.isfile(FLAGS.config_file):
    #     config = load_config(FLAGS.config_file)
    # else:
    #     print('qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq')
    #     config = config_model(char_to_id, tag_to_id)
    #     save_config(config, FLAGS.config_file)
    # make_path(FLAGS)
    config = config_model(char_to_id, tag_to_id)
    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        loss = []
        for i in range(100):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            if best:
                save_model(sess, model, FLAGS.ckpt_path, logger)
            evaluate(sess, model, "test", test_manager, id_to_tag, logger)


def evaluate_line(sent_s):
    print('start entity predict:')
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger, False)
        all_trained_char_embedding = model.char_lookup.eval()  # 已训练的字符向量
        sentence = sent_s.strip()  # sentence
        print('sentence:', sentence)
        result = model.evaluate_line(sess, input_from_line(sentence, char_to_id), id_to_tag)
        print(result)
        # 获取词向量
        all_entity = result['entities']
        if len(all_entity) > 0:
            for cur_qq in all_entity:
                start = cur_qq['start']
                end = cur_qq['end']
                embedding = all_trained_char_embedding[start: end, :]  # 获取词向量，start是词向量开始索引，end是结束索引:[,)
                embedding = np.sum(embedding, axis=0) / embedding.shape[0]  # 获取实体向量
            return embedding
        return None


def get_local_entity_embedding():
    # 获取本地实体的embedding，并保存
    # 获取方式：model.char_lookup训练的向量
    print('start entity local embedding:')
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        # 加载本地网络以及保存在 Detect/ckpt下的参数
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger, False)
        # 获取向量
        # print('model.char_lookup', model.char_lookup.eval())
        all_trained_char_embedding = model.char_lookup.eval()  # 已训练的字符向量
        all_data = open('../DataProcess/local_entity.txt', mode='r', encoding='utf-8').readlines()  # 读取知识点
        all_entity_local = dict()  # 保存本地实体的嵌入表示
        for cur_line in all_data:
            sentence = cur_line.strip()  # sentence
            result = model.evaluate_line(sess, input_from_line(sentence, char_to_id), id_to_tag)
            print('sentence:', result)
            # 获取词向量
            all_entity = result['entities']
            if len(all_entity) > 0:
                for cur_qq in all_entity:
                    word = cur_qq['word']
                    start = cur_qq['start']
                    end = cur_qq['end']
                    if word == result['string']:
                        embedding = all_trained_char_embedding[start: end, :]  # 获取词向量，start是词向量开始索引，end是结束索引:[,)
                        embedding = np.sum(embedding, axis=0) / embedding.shape[0]  # 获取实体向量
                        print('embedding:', embedding.shape)  # 打印维度
                        if word not in all_entity_local.keys():  # 相同的实体向量只保存1次
                            all_entity_local[word] = embedding  # 根据实体内容创建 实体-嵌入字典
        # local_entity_word_embedding已保存至本地，经测试local_entity_word_embedding已保存了实体向量
        pkl.dump(all_entity_local, open('../DataProcess/local_entity_word_embedding.pkl', mode="wb"))  # 将知识点的嵌入保存至本地


def main(_):
    # 首先训练train()，训练完后注释第265~272,运行get_local_entity_embedding，生成本地实体向量
    if FLAGS.train:
        if FLAGS.clean:
            clean(FLAGS)
        print('train the model.')
        train()
    else:
        print('evaluate')
        evaluate_line('结点，元素')
    # get_local_entity_embedding()


if __name__ == "__main__":
    tf.app.run(main)



