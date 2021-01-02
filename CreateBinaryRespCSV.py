import time
import pandas as pd
import numpy as np
import datatable as dtable

start = time.time()
print('Start Time = ', time.time() - start)

data = dtable.fread("[file path]/train.csv").to_pandas()
data = data.astype({c: np.float32 for c in data.select_dtypes(include='float64').columns})
data = data.query('weight > 0')
data = data.fillna(0)

data = data[['resp']]
x = [0 if i < 0 else 1 for i in data['resp']]

data['Binary Response'] = x
print(data.head())
data.to_csv('BinaryResp.csv')

print('Length = ', len(data))
print('Done = ', time.time() - start)
