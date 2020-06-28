import catboost
import pandas as pd
import os
from catboost import CatBoostClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import pymorphy2

if os.path.exists(os.path.join(os.path.abspath(os.curdir), 'models/model')):
    model = CatBoostClassifier()
    model.load_model('models/model')
    train_data = pd.read_excel('test.xlsx')
    comments = train_data['Comment']
    with open('models/model_file.txt', 'r') as file_model:
        dic_keys = file_model.read().split(',')
    data = np.zeros((len(comments), len(dic_keys) + 1))
    for i in range(0, len(comments)):
        for j in range(0, len(dic_keys)):
            if dic_keys[j] in comments[i]:
                data[i][j] = 1
            else:
                data[i][j] = 0
            data[i][-1] = len(comments[i])
    models_predict = model.predict(data=data)
    for i in range(len(models_predict)):
        print('{} : {}'.format(comments[i], models_predict[i]))

else:
    train_data = pd.read_excel('train.xlsx')
    comments = train_data['Comment']
    target = train_data['Toxic']
    normal_word = {}
    for i in comments:
        for word in i.split(' '):
            p = pymorphy2.MorphAnalyzer().parse(word=word)[0]
            if p.tag.POS not in ['NOUN', 'ADJF', 'ADJS', 'VERB', 'INFN', 'PRTF', 'PRTS', 'GRND', 'ADVB']:
                continue
            else:
                if p.normal_form in normal_word:
                    normal_word[p.normal_form] += 1
                else:
                    normal_word[p.normal_form] = 1
    dictionar = {}
    for key in normal_word.keys():
        if normal_word[key] > 5:
            dictionar[key] = normal_word.get(key)
        else:
            continue
    data = np.zeros((len(comments), len(dictionar) + 1))
    label = np.zeros((len(target)))
    dic_keys = list(dictionar.keys())
    for i in range(0, len(comments)):
        for j in range(0, len(dic_keys)):
            if dic_keys[j] in comments[i]:
                data[i][j] = 1
            else:
                data[i][j] = 0
            data[i][-1] = len(comments[i])
            label[i] = target[i]

    model = CatBoostClassifier()
    x_train, y_train, x_test, y_test = train_test_split(data, label, test_size=0.005)
    x_train.shape
    x_test.shape
    model.fit(data, label)
    print("Модель обучена, идет запись на носитель")
    np_correct = (y_test == model.predict(x_test)).sum()

    model.save_model('models/model',
                     format="cbm",
                     export_parameters=None,
                     pool=None)
    with open('models/model_file.txt', 'w') as file_model:
        file_model.write(','.join(dic_keys))
