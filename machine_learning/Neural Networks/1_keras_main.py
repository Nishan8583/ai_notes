from keras.models import Sequential  # Sequential construct neural layer sequentially on layer as layer basis
import numpy as np 
from keras.layers import Dense # Every single neuron is connected to every single neuron in the next layer

# XOR operation
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y= np.array([[0],[1],[1],[0]])
print(X,Y)

model = Sequential()
# input_dim=2 means previous layer has 2 neurons
# 4 means this layer has 4 neuron
# activation function is going to be sigmoid
# this is hidden layer
model.add(Dense(4,input_dim=2,activation="sigmoid"))
# this is output layer
model.add(Dense(1,input_dim=4,activation="sigmoid"))

#print(model.weights)

# loss= loss function
# optimizer, the function that adjusts weight value, like gradient descent
# metrics, measure the accuracy 
model.compile(loss="mean_squared_error",optimizer="adam",
              metrics=["binary_accuracy"])
# epochs is the number of iterations
model.fit(X,Y,epochs=10000,verbose=2)
print("FINISHED")
print(model.predict(X))