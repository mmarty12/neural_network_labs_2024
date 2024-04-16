import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = ['tench', 'English springer', 'cassette player', 'chainsaw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
epochs = 50
samples_num = 10

class AlexNet:
    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        model = models.Sequential([
            layers.Conv2D(filters=128, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(64, 64, 3)),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3, 3)),
            layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

class ImageClassifier:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = AlexNet().model
        self.train_dataset, self.validation_dataset = self.process_dataset()

    def process_image(self, image):
        image = tf.image.per_image_standardization(image)
        image = tf.image.resize(image, (64, 64))
        return image

    def process_dataset(self):
        train_dataset = self.dataset['train'].map(lambda image, label: (self.process_image(image), label)).batch(32)
        validation_dataset = self.dataset['validation'].map(lambda image, label: (self.process_image(image), label)).batch(32)
        return train_dataset, validation_dataset

    def train_model(self, epochs):
        history = self.model.fit(self.train_dataset, epochs=epochs, validation_data=self.validation_dataset, validation_freq=1)
        return history

def image_display(image, prediction):
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.xlabel(f'{prediction}')
    plt.imshow(image)
    plt.show()

def accuracy_graph_display(history):
    plt.plot(history.history['accuracy'], label='Accuracy', color='green')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def loss_graph_display(history):
    plt.plot(history.history['loss'], label='Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def main():
    data, info = tfds.load('imagenette/160px-v2', with_info=True, as_supervised=True)
    classifier = ImageClassifier(data)
    history = classifier.train_model(epochs)
    accuracy_graph_display(history)
    loss_graph_display(history)

    class_names = info.features['label'].names
    dataset_shuffled = data['validation'].shuffle(buffer_size=len(data['validation']))
    for _ in range(samples_num):
        image, label = next(iter(dataset_shuffled))
        processed_image = classifier.process_image(image[:])
        processed_image = tf.expand_dims(processed_image, axis=0)
        prediction = np.argmax(classifier.model.predict(processed_image)[0])

        image_display(image, CLASS_NAMES[prediction])

main()
