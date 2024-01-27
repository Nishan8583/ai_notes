import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# XOR is non linear, so thats why we do it.
train_data = np.array(
    [
        [0,0],
        [0,1],
        [1,0],
        [1,1],
    ],
    "float32",
)

target_data =np.array(
    [
        [0],
        [1],
        [1],
        [0],
    ],
    "float32",
)

# the new neural network
model = Sequential()
model.add(Dense(16,input_dim=2,activation="relu"))
model.add(Dense(16,input_dim=16,activation="relu"))
model.add(Dense(16,input_dim=16,activation="relu"))
model.add(Dense(16,input_dim=16,activation="relu"))
model.add(Dense(16,input_dim=16,activation="relu"))
model.add(Dense(16,input_dim=16,activation="relu"))
model.add(Dense(16,input_dim=16,activation="relu"))
model.add(Dense(1,input_dim=16,activation="sigmoid"))

# loss function is mean_sqaure_error, optimizer is adam 
model.compile(loss="mean_squared_error",
              optimizer="adam",
              metrics=["binary_accuracy"], # judging the performance of the network.
              )

# epoch is number of iteration 
model.fit(train_data,target_data,epochs=500,verbose=2)
print(model.predict(train_data).round())