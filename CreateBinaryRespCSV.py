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

y = data[['resp']]
x = [0 if i < 0 else 1 for i in y['resp']]
binary_df = pd.DataFrame({'Response':y, 'Classification':x})

binary_df.to_csv('BinaryResp.csv')

print('Length = ', len(y))
print('Done = ', time.time() - start)
