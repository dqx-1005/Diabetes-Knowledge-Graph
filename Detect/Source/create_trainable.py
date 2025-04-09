# coding=utf-8

all_data = open('../../processed/train_per_sentence.txt', mode='w', encoding='utf-8')
all_data_f = open('../../processed/adap_data.txt', mode='r', encoding='utf-8').readlines()
for f in all_data_f:
    f = f.strip().split('QAZQAZQAZ')
    for ff, fff in zip(f[0].split(), f[1].split()):
        all_data.write(' '.join([ff, fff]) + '\n')
    all_data.write('\n')

