import matplotlib.pyplot as plt
from keras.datasets import mnist  # contains 60k of handwritten images, 28x28 pixel image, 2x2 lists each having 28 elements
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator

(X_train, y_train),(X_test,y_test) = mnist.load_data()

# print("shape of train",X_train.shape)

#plt.imshow(X_train[0],cmap='gray')
#plt.title('Class '+str(y_train[0]))
#plt.show()

# Tensorflow handles the ormat (batch, height, width. channel (colors)
# ) 1 for grayscale, 3 for RGT, .shape[0] gives size
f_train = X_train.reshape(X_train.shape[0],28,28,1)
f_test = X_test.reshape(X_test.shape[0],28,28,1)

f_train=f_train.astype('float32')
f_test=f_test.astype('float32')

# normalize, min-max normalization, highest pixel intensity is 255
# will be within range 0,1
f_train /= f_train/255
f_test /= 255

# 10 output classes, but we need range of 0 and 1
# 0 = [1 0 0 0 0 0 0 0 0 0]
# 1 = [0 1 0 0 0 0 0 0 0 0]
# ....
t_train = to_categorical(y_train,10)
t_test = to_categorical(y_test,10)

print(t_test,t_train)

model = Sequential()

# number of filters (feature detectors) is 32, 3x3 is size of filter
# 28x28 input and 1 means just singal channel
# no need to specifiy the stride, default 1 pixel right and down
model.add(Conv2D(32,(3,3),input_shape=(28,28,1)))
model.add(Activation('relu'))

# after activation, use this normalization, maintains the mean activation close to 0
# and std close to 1, the scale of dimension remains the same, and reduces training time
model.add(BatchNormalization())

model.add(Conv2D(32,(3,2)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())

# regularization to avoid over-fitting
model.add(Dropout(0.3))
model.add(Dense(10,activation='softmax'))

#model.summary()


model.compile(loss="categorical_crossentropy",optimizer="adam",
              metrics=["accuracy"])
'''
model.fit(f_train,t_train,batch_size=128,epochs=5,
          validation_data=(f_test,t_test),verbose=1)

score=model.evaluate(f_test,t_test)
print("test accuracy %.2f"%score[1])

'''

train_generator = ImageDataGenerator(rotation_range=7, width_shift_range=0.05, shear_range=0.2,
                                     height_shift_range=0.07, zoom_range=0.05)
test_generator = ImageDataGenerator()

train_generator = train_generator.flow(f_train, t_train, batch_size=64)
test_generator = test_generator.flow(f_test, t_test, batch_size=64)

model.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=5,
                    validation_data=test_generator, validation_steps=10000//64)