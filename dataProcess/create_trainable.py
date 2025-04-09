# coding=utf-8

all_data = open('../processed/train_per_sentence.txt', mode='w', encoding='utf-8')
all_data_f = open('../processed/labeled.txt', mode='r', encoding='utf-8').readlines()
for f in all_data_f:
    f = f.strip().split('\t')
    if len(f) == 2:
        for ff, fff in zip(list(f[0]), f[1].split()):
            all_data.write(' '.join([ff, fff]) + '\n')
        all_data.write('\n')

