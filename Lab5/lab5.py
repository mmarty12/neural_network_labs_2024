import random
import cv2
import numpy as np
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.layers as layers
from keras.models import Model
from matplotlib import pyplot as plt

BREED_NAMES = ["Chihuahua", "Beagle", "Cocker Spaniel", "Old English Sheepdog", "Collie", "Bernese Mountain Dog"]
samples_num = 10
epochs = 15

def accuracy_graph_display(history):
    plt.plot(history.history['accuracy'], label='Accuracy', color='green')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def image_display(samples_num, prediction, testX, testY):
    fig = plt.figure(figsize=(20, 10))
    for i in range(samples_num):
      fig.add_subplot(samples_num // 2, 2, i + 1)
      image = testX[i][:, :, ::-1]
      plt.imshow(image)
      plt.axis("off")
      plt.title("Predicted: " + BREED_NAMES[np.argmax(prediction[i])] + " | Actual: " + BREED_NAMES[np.argmax(testY[i])])

    plt.show()

class ImageClassifier:
    def __init__(self, trainX, trainY, testX, testY):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY

        self.model = self.create_model(input_shape=trainX[0].shape, num_classes=trainY.shape[1])

    def create_model(self, input_shape, num_classes):
        base_model = InceptionV3(input_shape=input_shape, include_top=False, weights='imagenet')
        for layer in base_model.layers:
            layer.trainable = False

        x = layers.Flatten()(base_model.output)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(num_classes, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=output)
        return model

    def train_model(self, train_datagen, testX, testY, epochs=5):
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        history = self.model.fit(train_datagen.flow(self.trainX, self.trainY), validation_data=(testX, testY), epochs=epochs, validation_freq=1)
        return history

    def predict(self, data):
        return self.model.predict(data)

def main():
    dataPath = "dataset"
    data = []
    labels = []
    imagePaths = sorted(list(paths.list_images(dataPath)))
    random.seed(42)
    random.shuffle(imagePaths)

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (150, 150))
        data.append(image)
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
    enc = OneHotEncoder()
    trainY = trainY.reshape(-1, 1)
    testY = testY.reshape(-1, 1)
    trainY = enc.fit_transform(trainY).toarray()
    testY = enc.transform(testY).toarray()

    train_data = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

    classifier = ImageClassifier(trainX, trainY, testX, testY)
    history = classifier.train_model(train_data, testX, testY, epochs=epochs)
    prediction = classifier.predict(testX)

    accuracy_graph_display(history)
    image_display(samples_num, prediction, testX, testY)

main()
