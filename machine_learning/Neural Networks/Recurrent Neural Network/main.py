import numpy,pandas
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

NUM_OF_PREV_ITEMSS=5 # number of previous items to consider in the past for LSTM

numpy.random.seed(1)  # geenerate same random numbers

# load the dataset
# only use the second column
df= pandas.read_csv("daily_min_temperatures.csv",usecols=[1])
#print(df)
#print(df.values)

# Stationary time series can be predicted by RNN, if non stationary is not predictible
#plt.plot(df)
#plt.show()

# set type floating point
data = df.values.astype("float32")


# transform the vlaue into 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)

train, test = data[0:int(len(data) * 0.7), :], data[int(len(data) * 0.7):len(data), :]