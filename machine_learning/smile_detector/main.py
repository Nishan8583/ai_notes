from PIL import Image
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

pixels = []
labels=[]

# directory that contains the dataset
dir = "smiles_dataset/training_set/"

# populates the pixels and labes
def populate_features(pixels,labels):
    for filename in os.listdir(dir):
        image = Image.open(dir+filename).convert('1')
        pixels.append(list(image.getdata()))

        # one hot encoding happy (1,0), and sad (0,1)
        if filename.startswith("happy"):
            labels.append([1,0])
        else:
            labels.append([0,1])

populate_features(pixels,labels)

labels = np.array(labels)
pixels = np.array(pixels)

print(pixels.shape)
print(labels.shape)
pixels = pixels / 255.0  # normalizaing, transforming it into the range of 0 and 1


model = Sequential()
model.add(Dense(1024,input_dim=1024,activation="relu"))  # 32x32 = 1024
model.add(Dense(512,activation="relu"))
model.add(Dense(128,activation="relu"))
model.add(Dense(2,activation="softmax")) 

model.compile(loss="categorical_crossentropy",
              optimizer=Adam(lr=0.05),metrics=["accuracy"])

model.fit(pixels,labels,epochs=1000,batch_size=20,verbose=2)

print("Now testing")

test_pixel = []
test_labels = []
populate_features(test_pixel,test_labels)
print(model.predict(test_pixel))