# coding=utf-8
import os
import csv
import json
import pickle as pkl


class DataGet:
    def __init__(self):
        self.root = '../Source/2022_02_07/'

    def data_get(self):
        entity = csv.writer(open('../processed/entity.csv', mode='w', encoding='utf-8', newline=''))
        roles = csv.writer(open('../processed/roles.csv', mode='w', encoding='utf-8', newline=''))
        labeling = open('../processed/labeled.txt', mode='w', encoding='utf-8')
        entity.writerow(['entity:ID', 'name', ':LABEL'])
        roles.writerow([':START_ID', ':END_ID', ':TYPE'])
        names = list(map(lambda x: self.root + x, os.listdir(self.root)))
        print(names)
        entity_count = 0
        for file in names:
            file = json.load(open(file, mode='r', encoding='utf-8'))
            for pa in file['paragraphs']:
                for cur_sentence in pa['sentences']:
                    sen = cur_sentence['sentence']
                    label_bio = ['O'] * len(sen)
                    all_entity = cur_sentence['entities']
                    all_relations = cur_sentence['relations']
                    all_keys = dict()
                    for cur_entity in all_entity:
                        entity.writerow([cur_entity['entity_id'][0] + str(entity_count), cur_entity['entity'], cur_entity['entity_type']])
                        label_bio[cur_entity["start_idx"]] = 'B-' + cur_entity['entity_type']
                        if cur_entity["end_idx"] - cur_entity["start_idx"] > 1:
                            for j in range(cur_entity["start_idx"] + 1, cur_entity["end_idx"], 1):
                                label_bio[j] = 'I-' + cur_entity['entity_type']
                        entity_count += 1
                        if cur_entity['entity_id'] not in all_keys.keys():
                            all_keys[cur_entity['entity_id']] = cur_entity['entity']
                    for cur_relation in all_relations:
                        roles.writerow([all_keys[cur_relation['tail_entity_id']], all_keys[cur_relation['head_entity_id']],
                                        cur_relation['relation_type']])
                    labeling.write(sen + '\t' + ' '.join(label_bio) + '\n')

    def entity_clean(self):
        roles = csv.reader(open('../processed/roles.csv', mode='r', encoding='utf-8'))
        entity = csv.reader(open('../processed/entity.csv', mode='r', encoding='utf-8'))
        roles_cleaned = csv.writer(open('../processed/roles_cleaned.csv', mode='w', encoding='utf-8', newline=''))
        entity_cleaned = csv.writer(open('../processed/entity_cleaned.csv', mode='w', encoding='utf-8', newline=''))
        entity_cleaned.writerow(['entity:ID', 'name', ':LABEL'])
        roles_cleaned.writerow([':START_ID', ':END_ID', ':TYPE'])
        skip = True
        unique_all = []
        unique_entity = dict()
        for cur_role in roles:
            if skip:
                skip = False
            else:
                head_name, tail_name, en_type = cur_role[:]
                if [head_name, tail_name] not in unique_all:
                    unique_all.append([head_name, tail_name])
                    roles_cleaned.writerow([head_name, tail_name, en_type])
                else:
                    pass
        #
        skip = True
        for cur_entity in entity:
            if skip:
                skip = False
            else:
                entity_id, entity_name, entity_type = cur_entity[:]
                if entity_name not in unique_entity.keys():
                    unique_entity[entity_name] = 'T' + str(len(unique_entity.keys()))
                    entity_cleaned.writerow([unique_entity[entity_name], entity_name, entity_type])
                else:
                    pass
        pkl.dump(unique_entity, open('../processed/all_unique_entity.pkl', mode='wb'))

    def all_clean(self):
        roles = csv.reader(open('../processed/roles_cleaned.csv', mode='r', encoding='utf-8'))
        unique_entity = pkl.load(open('../processed/all_unique_entity.pkl', mode='rb'))
        roles_cleaned = csv.writer(open('../processed/roles_cleaned_cleaned.csv', mode='w', encoding='utf-8', newline=''))
        roles_cleaned.writerow([':START_ID', ':END_ID', ':TYPE'])
        skip = True
        for cur_entity in roles:
            if skip:
                skip = False
            else:
                en_id, en_id2, role_type = cur_entity[:]
                roles_cleaned.writerow([unique_entity[en_id], unique_entity[en_id2], role_type])
            # print('yes')

    def get_unique_entity(self):
        all_entity = csv.reader(open('../processed/entity_cleaned.csv', mode='r', encoding='utf-8'))
        entity = []
        for j in all_entity:
            if j[2].strip().__eq__('Disease'):
                if j[1].strip() not in entity:
                    entity.append(j[1].strip())
        print('entity:', entity)
        pkl.dump(entity, open('../processed/all_entity.pkl', mode='wb'))


if __name__ == "__main__":
    qwe = DataGet()
    qwe.data_get()
    qwe.entity_clean()
    qwe.all_clean()
    qwe.get_unique_entity()
