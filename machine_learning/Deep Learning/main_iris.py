from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

dataset = load_iris()

X = dataset.data
#print(X)
Y = dataset.target.reshape(-1,1) # make 2 dimensional 
#print(Y)

encoder = OneHotEncoder(sparse=False)
Y = encoder.fit_transform(Y)
# print(Y)
train_features, test_features, train_targets, test_targets = train_test_split(X, Y, test_size=0.2)
model = Sequential()
model.add(Dense(10,input_dim=4,activation="relu"))
model.add(Dense(10,input_dim=10,activation="relu"))
model.add(Dense(10,input_dim=10,activation="relu"))
model.add(Dense(3,input_dim=10,activation="softmax"))

optimizer = Adam(lr=0.05)
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"]
)

model.fit(train_features,train_targets,epochs=1000,batch_size=20,verbose=2)
results = model.evaluate(test_features,test_targets)
print("Accuracy = %.2f"%results[1])