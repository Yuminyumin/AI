import os
os.chdir('/Users/yuming/DataSet')

import ssl
import urllib.request

# SSL 인증서 검증을 무시하는 컨텍스트 생성
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import pandas as pd
#from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
#from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

train_dir = 'ani/train'
val_dir = 'ani/val'

data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                              horizontal_flip=True,
                                              width_shift_range=0.2,
                                              height_shift_range=0.2)
data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

image_size = 224

train_generator = data_generator_with_aug.flow_from_directory(
       directory=train_dir,
       target_size=(image_size, image_size),
       batch_size=12,
       class_mode='categorical')

validation_generator = data_generator_no_aug.flow_from_directory(
       directory=val_dir,
       target_size=(image_size, image_size),
       class_mode='categorical')


num_classes = 7

base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

cnn = Sequential()
cnn.add(base_model)
cnn.add(Flatten())
cnn.add(Dense(1024, activation='relu'))
cnn.add(Dense(num_classes, activation='softmax'))
cnn.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])

# define the checkpoint
filepath = "model/model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

hist = cnn.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator,
        callbacks=callbacks_list
        )

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

#완료 62프로