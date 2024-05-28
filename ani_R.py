import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import regularizers
import os
import matplotlib.pyplot as plt

train_folder = '/Users/yuming/DataSet/ani/train'
test_folder = '/Users/yuming/DataSet/ani/val'

class_reduce = 0.5
no_class = int(len([d for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d))]) * class_reduce)

x_train, y_train = [], []
for i, class_name in enumerate(os.listdir(train_folder)):
    class_path = os.path.join(train_folder, class_name)
    if os.path.isdir(class_path) and i < no_class:
        for fname in os.listdir(class_path):
            img_path = os.path.join(class_path, fname)
            img = image.load_img(img_path, target_size=(224, 224))
            if len(img.getbands()) != 3:
                print("주의: 유효하지 않은 영상 발생", class_name, fname)
                continue
            x = image.img_to_array(img)
            x = preprocess_input(x)
            x_train.append(x)
            y_train.append(i)

x_test, y_test = [], []
for i, class_name in enumerate(os.listdir(test_folder)):
    class_path = os.path.join(test_folder, class_name)
    if os.path.isdir(class_path) and i < no_class:
        for fname in os.listdir(class_path):
            img_path = os.path.join(class_path, fname)
            img = image.load_img(img_path, target_size=(224, 224))
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

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
cnn = Sequential()
cnn.add(base_model)
cnn.add(Flatten())
cnn.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)))  # L2 규제 추가
cnn.add(Dropout(0.5))
cnn.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))  # L2 규제 추가
cnn.add(BatchNormalization())
cnn.add(Dense(no_class, activation='softmax'))

cnn.compile(loss='categorical_crossentropy', optimizer=Adam(0.00002), metrics=['accuracy'])
hist = cnn.fit(x_train, y_train, batch_size=16, epochs=10, validation_data=(x_test, y_test), verbose=1)

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

#완료 정확도 0.3 