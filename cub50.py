import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

train_folder = '/Users/yuming/DataSet/CUB200/train'
test_folder = '/Users/yuming/DataSet/CUB200/val'

class_reduce = 0.1
no_class = int(len(os.listdir(train_folder)) * class_reduce)

x_train, y_train = [], []
for i, class_name in enumerate(os.listdir(train_folder)):
    if i < no_class:
        for fname in os.listdir(train_folder + '/' + class_name):
            img = image.load_img(train_folder + '/' + class_name + '/' + fname, target_size=(224, 224))
            if len(img.getbands()) != 3:
                print("주의: 유효하지 않은 영상 발생", class_name, fname)
                continue
            x = image.img_to_array(img)
            x = preprocess_input(x)
            x_train.append(x)
            y_train.append(i)

x_test, y_test = [], []
for i, class_name in enumerate(os.listdir(test_folder)):
    if i < no_class:
        for fname in os.listdir(test_folder + '/' + class_name):
            img = image.load_img(test_folder + '/' + class_name + '/' + fname, target_size=(224, 224))
            if len(img.getbands()) != 3:
                print("주의: 유효하지 않은 영상 발생", class_name, fname)
                continue
            x = image.img_to_array(img)
            x = preprocess_input(x)
            x_test.append(x)
            y_test.append(i)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
y_train = tf.keras.utils.to_categorical(y_train, no_class)
y_test = tf.keras.utils.to_categorical(y_test, no_class)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(x_train)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
cnn = Sequential()
cnn.add(base_model)
cnn.add(Flatten())
cnn.add(Dense(1024, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(no_class, activation='softmax'))

cnn.compile(loss='categorical_crossentropy', optimizer=Adam(0.00002), metrics=['accuracy'])
hist = cnn.fit(datagen.flow(x_train, y_train, batch_size=16), epochs=10, validation_data=(x_test, y_test), verbose=1)

res = cnn.evaluate(x_test, y_test, verbose=0)
print("정확률은", res[1] * 100)

# 정확률 그래프
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.grid()
plt.show()

# 손실 함수 그래프
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.grid()
plt.show()
