import glob
import os
from sklearn.model_selection import train_test_split
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy
import numpy as np
from os import listdir
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
from keras.models import model_from_json
import h5py
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

path = 'C:/Users/Mihai/PycharmProjects/p2/defecte/'
path1 = 'C:/Users/Mihai/PycharmProjects/p2/non-defecte/'
batch_size = 6
num_classes = 2
epochs = 100
data_augumentation = True
paths = []
dataArray = []
paths.append(path)
paths.append(path1)

j = 0
#aksh = np.empty((0))
for i in paths:
    imagesList = listdir(i)
    # Imaginile trebuie sa fie de aceeasi dimensiune. La el cred ca erau 256x256
    for image in imagesList:
        img = Image.open(i + image)
        im = (np.array(img, ndmin=3))
        r = im[:,:,0].flatten()
        g = im[:,:,1].flatten()
        b = im[:,:,2].flatten()
        label = [j]
        # out = r.tolist() + g.tolist() + b.tolist() + label
        out = r.tolist()[0:100] + g.tolist()[0:100] + b.tolist()[0:100] + label
        #aksh = np.concatenate((aksh, out), axis=0)
        dataArray.append(out)
    j = j + 1

myarray = np.asarray(dataArray)

Y = myarray[:, -1]
X = myarray[:, 0:-1]

newx = []

for r in X:
    # newvec = r.reshape(200, 200, 3)
    newvec = r.reshape(10, 10, 3)
    img = Image.fromarray(newvec, 'RGB')
    newx.append(newvec)

finalarray = np.array(newx)
# print((finalarray.shape))


x_train, x_test, y_train, y_test = train_test_split(finalarray, Y, test_size=0.1)
y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))

# initializam  optimizatorul RMSprop
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# antrenam modelul folosind RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augumentation:
    print('Not using data augumentation')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augumentation.')
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False)
    datagen.fit(x_train)

    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))

model_json = model.to_json()
with open("model,json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved module to disk")
