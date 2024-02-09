import numpy,pandas
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error


NUM_OF_PREV_ITEMSS=5 # number of previous items to consider in the past for LSTM

# create a 2D array, basically get n data, and then next will be the prediction
# again get next 5 and so on
#[1,2,3,4,5,6][1,2], first [1,2,3,4,5][1] next [2,3,4,5,6][2]
def reconstruct(data_set, n=1):
    x, y = [], []

    for i in range(len(data_set) - n - 1):
        a = data_set[i:(i + n), 0]
        x.append(a)
        y.append(data_set[i + n, 0])

    return numpy.array(x), numpy.array(y)



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

#(seventy percent is training, 30 is for test)
train, test = data[0:int(len(data) * 0.7), :], data[int(len(data) * 0.7):len(data), :]

train_x, train_y= reconstruct(train,NUM_OF_PREV_ITEMSS)
test_x, test_y=reconstruct(test,NUM_OF_PREV_ITEMSS)

# reshape input to be [numOfSamples, time steps, numOfFeatures]
# time steps is 1 because we want to predict the next value (t+1)
train_x = numpy.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
test_x = numpy.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

# retrun_sequences = TRUE because we want to return a sequence, because this is going
# to be input of next layer
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(1, NUM_OF_PREV_ITEMSS)))
model.add(Dropout(0.5)) # omit a single nueron with 50% probablity, prevent overfitting
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=50)) # last layer, so we dont need to return sequence
model.add(Dropout(0.3))
# Upto now 3 RNN layer

model.add(Dense(units=1))  # Finaly we need one Dense layer, units =1, becasue its goint to be scalaer


model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_x, train_y, epochs=10, batch_size=16, verbose=2)

# make predictions and min-max normalization
test_predict = model.predict(test_x)
test_predict = scaler.inverse_transform(test_predict) # inverse_transform because we applied Min-Max Normalization, so wee need original data
test_labels = scaler.inverse_transform([test_y])

test_score = mean_squared_error(test_labels[0], test_predict[:, 0])
print('Score on test set: %.2f MSE' % test_score)

# plot the results (original data + predictions)
test_predict_plot = numpy.empty_like(data)
test_predict_plot[:, :] = numpy.nan
test_predict_plot[len(train_x)+2*NUM_OF_PREV_ITEMSS+1:len(data)-1, :] = test_predict
plt.plot(scaler.inverse_transform(data))

