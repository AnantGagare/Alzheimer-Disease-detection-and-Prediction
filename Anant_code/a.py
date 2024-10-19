import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle  # New import for shuffle
from keras.utils import to_categorical  # Updated import for to_categorical




# Image dimensions
img_rows, img_cols = 64, 64
img_channels = 3

# Paths to the image directories
path1 = 'D:/All_Code/Code/Alzheimer_Disease/train/4'
path2 = 'D:/All_Code/Code/Alzheimer_Disease/training_set/2'

# Load the images from path1, resize, and save them in path2
listing = os.listdir(path1)
num_samples = len(listing)
print(f'Number of samples: {num_samples}')

for file in listing:
    img = Image.open(os.path.join(path1, file)).resize((img_rows, img_cols)).convert('RGB')
    img.save(os.path.join(path2, file), "png")

# Shuffle and split the data into train and test sets
data = []
labels = []

for file in os.listdir(path2):
    img = np.array(Image.open(os.path.join(path2, file)))
    data.append(img)
    labels.append(0 if 'class_0' in file else 1)  # Assuming class labeling is part of file name

data = np.array(data)
labels = np.array(labels)

data, labels = shuffle(data, labels, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Reshape and normalize the data
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels).astype('float32') / 255

# Convert class labels to categorical (one-hot encoding)
y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)

