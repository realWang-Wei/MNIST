from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import numpy as np
import time

def run():
    # models
    names = ['KNN', 'Adaboost', 'RF', 'DT', 'SVC' ]
    models = [ KNeighborsClassifier(), AdaBoostClassifier(), RandomForestClassifier(), DecisionTreeClassifier(), SVC()]

    # prepare
    PATH = '../data/'
    OUT_PATH = '../output/'
    train = pd.read_csv(PATH + 'train.csv')
    X_test = pd.read_csv(PATH + 'test.csv')
    sample = pd.read_csv(PATH + 'sample_submission.csv')
    
    y_train = train.label
    X_train = train.drop(['label'], axis=1)
    
    for i in range(len(models)):
        model = models[i]
        # train
        start = time.time()
        model.fit(X_train, y_train)
        finish = time.time()
        print('Time Cost of {} : {}'.format(names[i], finish-start))

        # predict
        predictions = model.predict(X_test)

        # results = np.argmax(predictions,axis = 1)

        df = {'ImageId':sample['ImageId'],
             'Label':predictions }

        submission = pd.DataFrame(df)

        submission.to_csv(OUT_PATH + names[i] + '.csv', index=False)

if __name__ == '__main__':
    run()