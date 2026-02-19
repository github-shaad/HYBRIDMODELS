import pandas as pd 
import numpy as np

df = pd.read_excel("RawData.xlsx", header=None)

tickers = df.iloc[1:, 0].to_numpy().astype(str)    # First column only
time_series = df.iloc[1:, 1:].replace({'\$': '', ',': '', }, regex=True).astype(float).to_numpy()  
log_time_series = np.log(time_series)

relative_time_series = np.array([])
chunk = []
for i,element in enumerate(log_time_series):
    row = []
    for j,e in enumerate(element):
        if j == 0:
            row.append(element[j])
        else:
            row.append(element[j] - element[j-1])
    chunk.append(row)
relative_time_series = np.vstack((chunk))



data = relative_time_series

num_time_steps = data.shape[1]
split_idx = int(num_time_steps * 0.8)

train_data = data[:, :split_idx]
test_data  = data[:, split_idx:]
print(tickers)
np.save("raw_series.npy", time_series)
np.save("tickers.npy", tickers)
np.save("data_diff.npy", data)
np.save("train_data_diff.npy", train_data)
np.save("test_data_diff.npy", test_data)