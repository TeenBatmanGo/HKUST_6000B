
from scipy.misc import imread, imresize
import numpy as np
import pandas as pd
from keras.applications import ResNet50
from keras.preprocessing import image
from keras.models import Sequential, Model, Input
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.utils import np_utils


train = pd.read_table('train.txt', sep=' ', header=None)
train.columns = ['path', 'label']
val = pd.read_table('val.txt', sep=' ', header=None)
val.columns = ['path', 'label']
test = pd.read_table('test.txt', sep=' ', header=None)
test.columns = ['path']


def img_resize(img, size=200):
    img_resized = imresize(img, (size, size))
    return img_resized


height, width, depth = img_resize(img_li[0], 224).shape
num_train = len(train)
num_val = len(val)
num_test = len(test)
num_classes = len(np.unique(train.label.values))
X_train = np.zeros((num_train, height, width, depth))
X_val = np.zeros((num_val, height, width, depth))
X_test = np.zeros((num_test, height, width, depth))
y_train = train.label.values.reshape((-1, 1))
y_val = val.label.values.reshape((-1, 1))
y_train = np_utils.to_categorical(y_train, num_classes)
y_val = np_utils.to_categorical(y_val, num_classes)


for i in range(num_train):
    img = imread(train.iloc[i, :].path)
    img = img_resize(img, 224)
    X_train[i, :, :, :] = img
    
for i in range(num_val):
    img = imread(val.iloc[i, :].path)
    img = img_resize(img, 224)
    X_val[i, :, :, :] = img
    
for i in range(num_test):
    img = imread(test.iloc[i, :].path)
    img = img_resize(img, 224)
    X_test[i, :, :, :] = img


base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=Input((224, 224, 3)))
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = Flatten(name='flatten')(x)
x = Dropout(0.6)(x)
# x = Dense(64, activation='relu', name='fc1')(x)
x = Dense(5, activation='softmax', name='predictions')(x)


model = Model(inputs=base_model.input, output=x)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=64, nb_epoch=50, validation_split=0.2)
preds_result = model.predict(X_test)
output = np.argmax(preds_result, axis=1)


with open('test2.txt', 'w+') as f:
    cnt = 0
    for line in test.values:
        line1 = line[0] + ' ' + str(output[cnt]) + '\n'
        cnt += 1
        f.write(line1)