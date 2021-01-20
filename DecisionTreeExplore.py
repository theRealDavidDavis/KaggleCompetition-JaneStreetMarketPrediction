import time
import pandas as pd
import numpy as np
import datatable as dtable

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

start = time.time()
print('Start Time = ', time.time() - start)

data = dtable.fread("C:/Users/p2dav/Desktop/Data Science and Machine Learning/Kaggle Competitions/2021/Jane Street Market Prediction/train.csv").to_pandas()
data = data.drop(['date', 'ts_id'], axis=1)
data = data.query('weight > 0')
data = data.fillna(0)

# To test changes, adjust the number of rows used from the data file. This script is built to support loading less rows.
# data = data.head(50000)

options = np.arange(1, 4, 1)
data_length = (len(data) // 10000) * 10000
total_count = np.arange(10000, data_length, 10000)
sizes = np.array([.25, .3, .35, .4, .45, .5, .55, .6])
scores = np.empty((0, 1), int)
accuracies = np.empty((0, 1), int)
track_rows = np.empty((0, 1), int)
track_sizes = np.empty((0, 1), int)
track_seq = np.empty((0, 1), int)

for seq in options:
    for rows in total_count:
        print('Seq = ', seq)
        print('# Rows = ', rows)
        if seq == 1:
            head_count = rows
            data = data.head(head_count)
        elif seq == 2:
            tail_count = rows
            data = data.tail(tail_count)
        elif seq == 3:
            if ((rows / 10000) % 2) == 0:
                head_count = rows//2
                tail_count = rows//2
                dataTail = data.tail(tail_count)
                data = data.head(head_count)
                data.append(dataTail)

        print('DataFrame ready at time', time.time() - start)
        x = data.drop(['resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp'], axis=1)
        y = data[['resp']]

        for sz in sizes:
            x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=sz, test_size=.39)
            model = tree.DecisionTreeRegressor()
            model.fit(x_train, y_train)
            print('train size = ', sz)
            print('Model fitted at time', time.time() - start)

            y_predict = model.predict(x_test)
            print('Predictions made at time', time.time() - start)

            # Get the Score
            tmpS = model.score(x_test, y_test)
            scores = np.append(scores, tmpS)

            z_predict = [0 if i < 0 else 1 for i in y_predict]
            z_test = [0 if i < 0 else 1 for i in y_test['resp']]

            tmpA = accuracy_score(z_test, z_predict)
            accuracies = np.append(accuracies, tmpA)

            track_seq = np.append(track_seq, seq)
            track_rows = np.append(track_rows, rows)
            track_sizes = np.append(track_sizes, sz)

            print('Score = ', tmpS)
            print('Accuracy =', tmpA)

    tmpData = pd.DataFrame({'Score': scores, 'Accuracy': accuracies, 'Seq': track_seq, 'Rows': track_rows, 'Train Size': track_sizes})
    filename = 'JaneStreet_DTM_Explore_Seq_' + str(seq) + '_Rows_' + str(rows) + '_size_' + str(sz) + '.csv'
    tmpData.to_csv(filename)
    print('Total Time for seq loop', time.time() - start)

print('Total Time for script', time.time() - start)
