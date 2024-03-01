import numpy as np

class Perceptron:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize random weights for input to hidden layer and hidden to output layer
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)

    def activate_func(self, x):
        # Step activation function: Returns 1 if x > 0, else returns 0
        return np.where(x > 0, 1, 0)

    def train(self, inputs, outputs, epochs, learning_rate):
        for _ in range(epochs):
            # Compute the output of hidden layer
            hidden_layer_output = self.activate_func(np.dot(inputs, self.weights_input_hidden))
            predicted_output = self.activate_func(np.dot(hidden_layer_output, self.weights_hidden_output))

            # Compute the errors and delta for the hidden and output layers
            output_error = outputs - predicted_output
            output_delta = output_error * 1

            hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
            hidden_layer_delta = hidden_layer_error * 1

            # Update weights
            self.weights_hidden_output += learning_rate * hidden_layer_output.T.dot(output_delta)
            self.weights_input_hidden += learning_rate * inputs.T.dot(hidden_layer_delta)

    def predict(self, inputs):
        # Make predictions using the trained perceptron
        hidden_layer_output = self.activate_func(np.dot(inputs, self.weights_input_hidden))
        return self.activate_func(np.dot(hidden_layer_output, self.weights_hidden_output))

# XOR input and output for further testing
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_data = np.array([[0], [1], [1], [0]])

# Number of epochs and learning rate
epochs = 150
learning_rate = 0.1

# Size of layers
input_size = 2
hidden_size = 2
output_size = 1

# Create and train the neural network
perceptron = Perceptron(input_size, hidden_size, output_size)
perceptron.train(input_data, output_data, epochs, learning_rate)

# Predict and print results
for test_input, target_output in zip(input_data, output_data):
    training_res = perceptron.predict(test_input)[0]
    print(f"{test_input} -> {training_res} | Target result: {target_output[0]}")
