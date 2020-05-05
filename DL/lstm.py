from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.utils import np_utils

import pandas as pd
import numpy as np
import time

def Net(n_inputs=28, time_steps=28, n_units=28, out_dim=10):
    model = Sequential()
    model.add(LSTM(n_units, input_shape=(time_steps, n_inputs)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(out_dim, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


EPOCHS = 32
BATCH_SIZE = 32
PATH = '../data/'
OUT_PATH = '../output/'

if __name__ == '__main__':
    train_df = pd.read_csv(PATH + 'train.csv')
    test = pd.read_csv(PATH + 'test.csv');test = np.array(test)
    sample = pd.read_csv(PATH + 'sample_submission.csv')

    X_train = train_df.drop(['label'], axis=1);X_train = np.array(X_train)
    X_train = X_train.reshape((-1, 28, 28))
    y_train = train_df.label
    y_train = np_utils.to_categorical(y_train, 10)

    lstm = Net()
    start = time.time()
    lstm.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=1)
    finish = time.time()
    print('Time Cost:', finish - start)
    
    X_test = test.reshape((-1, 28, 28))
    predictions = lstm.predict(X_test)

    results = np.argmax(predictions,axis = 1)

    df = {'ImageId':sample['ImageId'],
         'Label':results }

    submission = pd.DataFrame(df)

    submission.to_csv(OUT_PATH + 'LSTM.csv', index=False)