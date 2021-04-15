import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

keras = tf.keras


def loading_data():
    dirname = 'dataset/'
    dirtype = ['train/', 'valid/']
    dirclass = ['mask/', 'nomask/']

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for typ in dirtype:
        for cls in dirclass:
            for i in os.listdir(dirname + typ + cls):
                if typ == 'train/':
                    x_train.append(dirname + 'train/' + cls + i)
                    y_train.append(cls[:-1])
                else:
                    x_test.append(dirname + 'valid/' + cls + i)
                    y_test.append(cls[:-1])
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


x_train, y_train, x_test, y_test = loading_data()


def process_label(label):
    label = [i == unique_label for i in label]
    label = np.array(label).astype(int)
    return label


def processImage(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[224, 224])
    return image


def pairData(image, label):
    return processImage(image), label


def batchData(image, label=None, for_valid=False, for_test=False):
    if for_test:
        data = tf.data.Dataset.from_tensor_slices((image))
        batch = data.map(processImage).batch(16)
        return batch
    elif for_valid:
        data = tf.data.Dataset.from_tensor_slices((tf.constant(image), tf.constant(label)))
        batch = data.map(pairData).batch(16)
        return batch
    else:
        data = tf.data.Dataset.from_tensor_slices((tf.constant(image), tf.constant(label)))
        data = data.shuffle(buffer_size=len(image))
        batch = data.map(pairData).batch(16)
        return batch


unique_label = np.unique(y_test)
y_test = process_label(y_test)
y_train = process_label(y_train)

train_data = batchData(x_train, y_train)
valid_data = batchData(x_test, y_test, for_valid=True)


# # Начало обучения
# model = keras.Sequential([
#     keras.layers.Conv2D(input_shape=(224, 224, 3), filters=32, kernel_size=(3, 3), activation='relu'),
#     keras.layers.MaxPooling2D(),
#
#     keras.layers.Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), activation='relu'),
#     keras.layers.MaxPooling2D(),
#     keras.layers.Conv2D(input_shape=(224, 224, 3), filters=128, kernel_size=(3, 3), activation='relu'),
#     keras.layers.MaxPooling2D(),
#     keras.layers.Flatten(),
#     keras.layers.Dense(128),
#     keras.layers.Activation('relu'),
#
#     keras.layers.Dense(2),
#     keras.layers.Activation('softmax')
# ])
#
# model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['acc'])
#
# early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
#
# history = model.fit(train_data, validation_data=valid_data, validation_freq=1, epochs=70, verbose=1, )
# model.save('complete_model.h5')
# #Конец обучения


def resolve(pred):
    mask = pred[0]
    nomask = pred[1]
    if mask > nomask:
        return "Человек на фото в маске с вероятностью - " + str(round(mask * 100, 2)) + "%"
    else:
        return "Человек на фото без маски с вероятностью - " + str(round(nomask * 100, 2)) + "%"


def predict(unseen_image=[]):
    model = keras.models.load_model('complete_model.h5')  # Loading Saved NN Model
    test_data = batchData(unseen_image, for_test=True)
    prediction = model.predict(test_data)

    for image, pred in zip(unseen_image, prediction):
        result = resolve(pred)
        fig, axe = plt.subplots()
        axe.imshow(processImage(image))
        axe.axis(False)
        axe.set_title(result)
        plt.show()
    shutil.rmtree('dataset/test')
    os.makedirs('dataset/test')
