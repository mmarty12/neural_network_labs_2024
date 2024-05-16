import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models
from tensorflow.keras.layers import LSTM, Dense, Embedding
import matplotlib.pyplot as plt
import numpy as np

test_sentences = [
    "The packaging was impressive, and the item arrived in perfect condition.", # +
    "The characters were well-defined, but lacked depth and complexity.",       # +-
    "The application keeps crashing, it's very frustrating to use.",            # -
    "The play was fantastic, the actors gave a stellar performance.",           # +
    "The plot was straightforward, with no unexpected twists."                  # +-
]

def accuracy_graph_display(history):
    plt.plot(history.history['accuracy'], label='Accuracy', color='green')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

class SentimentAnalysisModel:
    def __init__(self, vocabulary=20000, length=200):
        self.vocabulary = vocabulary
        self.length = length
        self.tokenizer = None
        self.model = None

    def prepare_data(self, trainX):
        self.tokenizer = Tokenizer(num_words=self.vocabulary)
        self.tokenizer.fit_on_texts(trainX)

    def create_model(self):
        self.model = models.Sequential([
            Embedding(self.vocabulary, 8),
            LSTM(16),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid")
        ])

        self.model.compile(optimizer="RMSprop", loss="binary_crossentropy", metrics=["accuracy"])
        return self.model

    def train_model(self, trainX, trainY, testX, testY, epochs=10):
        train_t = self.tokenizer.texts_to_sequences(trainX)
        test_t = self.tokenizer.texts_to_sequences(testX)
        train_t = pad_sequences(train_t, maxlen=self.length)
        test_t = pad_sequences(test_t, maxlen=self.length)
        history = self.model.fit(train_t, np.array(trainY), epochs=epochs, validation_data=(test_t, np.array(testY)))

        return history

    def predict_sentiment(self, sentences):
        for sentence in sentences:
            print(sentence)
            input_seq = pad_sequences(self.tokenizer.texts_to_sequences([sentence]), maxlen=self.length)
            prediction = self.model.predict(input_seq, verbose=0)
            prediction_score = prediction[0][0]
            if prediction_score < 0.45:
                sentiment = 'Negative'
            elif prediction_score > 0.85:
                sentiment = 'Positive'
            else:
                sentiment = 'Neutral'
            print(f"Prediction: {sentiment} ({prediction_score})")

def main():
    data = tfds.load("yelp_polarity_reviews", as_supervised=True)
    train_set, test_set = data['train'], data['test']
    trainX, trainY = [], []
    for element in train_set:
        trainX.append(element[0].numpy().decode())
        trainY.append(int(element[1].numpy()))

    testX, testY = [], []
    for element in test_set:
        testX.append(element[0].numpy().decode())
        testY.append(int(element[1].numpy()))

    model = SentimentAnalysisModel()
    model.prepare_data(trainX)
    model.create_model()
    history = model.train_model(trainX, trainY, testX, testY)
    accuracy_graph_display(history)

    model.predict_sentiment(test_sentences)

main()
