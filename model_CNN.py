from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from numpy import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Input image dimensions
img_rows, img_cols = 64, 64
img_channels = 3  # Number of channels

# Path of folder of images
path1 = 'D:/All Code/Code/Alzheimer_Disease/Alzheimer_Disease/Alzheimer_Disease/train/4'
path2 = 'D:/All Code/Code/Alzheimer_Disease/Alzheimer_Disease/Alzheimer_Disease/training_set/2'

listing = os.listdir(path1)
num_samples = size(listing)
print(num_samples)

for file in listing:
    im = Image.open(path1 + '\\' + file)
    img = im.resize((img_rows, img_cols))
    gray = img.convert(mode='RGB')
    gray.save(path2 + '\\' + file, "png")

imlist = os.listdir(path2)

# Image pre-processing and label assignment
immatrix = array([array(Image.open(path2 + '/' + im2)).flatten() for im2 in imlist], 'f')

label = np.ones((num_samples,), dtype=int)
label[0:245] = 0  # For class 0
label[245:288] = 1  # For class 1

# Shuffle the data
data, Label = shuffle(immatrix, label, random_state=2)
train_data = [data, Label]

# Splitting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(data, Label, test_size=0.2)

# Reshaping the data
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalizing the pixel values
X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# One-hot encoding the labels
nb_classes = 7  # Number of classes
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

# Building the model
model = Sequential()

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, img_channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes, activation='softmax'))

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

# Training the model
batch_size = 32
nb_epoch = 10  # You can increase this for better results

hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

# Evaluating the model
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Saving the model
model.save('best_model.h5')

# Visualizing the loss and accuracy
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(nb_epoch)

plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Training Loss vs Validation Loss')
plt.grid(True)
plt.legend(['Train', 'Validation'])
plt.show()

plt.figure(2, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy vs Validation Accuracy')
plt.grid(True)
plt.legend(['Train', 'Validation'], loc=4)
plt.show()

# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)

target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6']
print(classification_report(np.argmax(Y_test, axis=1), y_pred, target_names=target_names))
print(confusion_matrix(np.argmax(Y_test, axis=1), y_pred))

# Saving the model weights
model.save_weights('weights-Test-CNN.h5')
