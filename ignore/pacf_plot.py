import numpy as np 
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

data = np.load("train_data.npy")

stock = data[0]

plt.figure(figsize=(20,10))

plot_pacf(stock, lags = 60, method="ywm", ax = plt.gca())
plt.show()