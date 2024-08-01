from tkinter.tix import MAX
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.models import Model
from keras.layers import Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
import pprint
import sys
import pygad
import random

sys.displayhook = pprint.pprint

LEARNING_RATE = 0.01
# IRIS dataset
MAX_LAYERS = 10
MAX_LAYER_SIZE = 24
INPUT_LAYER_SIZE = 4
OUTPUT_LAYER_SIZE = 3

# # MNIST dataset
# MAX_LAYERS = 10
# MAX_LAYER_SIZE = 512
# INPUT_LAYER_SIZE = 28 * 28
# OUTPUT_LAYER_SIZE = 10

ACTIVATIONS = {1: 'relu', 2: 'sigmoid', 3: 'tanh'}
ACTIVATIONS_OUTPUT = {1: 'softmax'}


class NeuralNetwork():
    def __init__(self, num_layers, hidden_layer_sizes, activations, dropout_rates, batch_norms, activation_output):
        self.num_layers = num_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activations = activations
        self.dropout_rates = dropout_rates
        self.batch_norms = batch_norms
        self.activation_output = activation_output
        
        self.create_model()
        
    def create_model(self):
        # Define the input layer
        input_layer = Input(shape=(INPUT_LAYER_SIZE,), name='input')

        # Define the hidden layers
        first = True
        hidden_layer = None
        for i in range(0, self.num_layers):            
            if first:
                hidden_layer = Dense(self.hidden_layer_sizes[i], name='hidden_'+str(i + 1))(input_layer)
                if self.batch_norms[i] == 1:
                    hidden_layer = BatchNormalization()(hidden_layer)
                hidden_layer = Activation(self.activations[i])(hidden_layer)
                if self.dropout_rates[i] > 0:
                    hidden_layer = Dropout(self.dropout_rates[i])(hidden_layer)
                first = False
            else:
                hidden_layer = Dense(self.hidden_layer_sizes[i], name='hidden_'+str(i + 1))(hidden_layer)
                if self.batch_norms[i] == 1:
                    hidden_layer = BatchNormalization()(hidden_layer)
                hidden_layer = Activation(self.activations[i])(hidden_layer)
                if self.dropout_rates[i] > 0:
                    hidden_layer = Dropout(self.dropout_rates[i])(hidden_layer)

        # Define the output layer
        output_layer = Dense(OUTPUT_LAYER_SIZE, name='output')(hidden_layer)
        if self.batch_norms[-1] == 1:
            output_layer = BatchNormalization()(output_layer)
        output_layer = Activation(self.activation_output)(output_layer)

        # Create the model
        self.model = Model(inputs=input_layer, outputs=output_layer)

        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Summary of the model
        # self.model.summary()


def array_to_nn(ga_array):
    # print("————————————————————> GA_ARRAY: ", ga_array)
    num_layers = ga_array[0]
    hidden_layer_sizes = ga_array[1:MAX_LAYERS + 1]
    activations_keys = ga_array[MAX_LAYERS + 1:2 * MAX_LAYERS + 1]
    dropout_rates = ga_array[2 * MAX_LAYERS + 1:3 * MAX_LAYERS + 1]
    batch_norms = ga_array[3 * MAX_LAYERS + 1:4 * MAX_LAYERS + 1]
    activation_output_key = ga_array[4 * MAX_LAYERS + 1]
    
    activations = [ACTIVATIONS[int(key)] if int(key) != -1 else key for key in activations_keys]  # Ignore -1 as padding
    activation_output = ACTIVATIONS_OUTPUT[int(activation_output_key)] if activation_output_key != -1.0 else activation_output  # Ignore -1 as padding
    
    nn = NeuralNetwork(int(num_layers), hidden_layer_sizes.astype(np.int32), activations, dropout_rates, batch_norms.astype(np.int32), activation_output)
    
    return nn


def plot_data(dataset):
    _, ax = plt.subplots()
    scatter = ax.scatter(dataset.data[:, 0], dataset.data[:, 1], c=dataset.target)
    ax.set(xlabel=dataset.feature_names[0], ylabel=dataset.feature_names[1])
    _ = ax.legend(scatter.legend_elements()[0], dataset.target_names, loc="lower right", title="Classes")
    plt.show()

if __name__ == '__main__':
    # # Fix random seed for reproducibility
    # rnd_seed = 42
    # np.random.seed(rnd_seed)
    
    # # Load the Iris dataset
    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target

    # plot_data(iris)

    # # One-hot encode the labels
    # y = to_categorical(y)

    # # Split the dataset into a training set and a test set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Create the neural network
    # nn = NeuralNetwork(activations=['relu' * MAX_LAYERS], hidden_layer_sizes=[4, 32, 3])

    # # Train the model
    # nn.model.fit(X_train, y_train, epochs=100, batch_size=32)

    # # Evaluate the model on the training data
    # train_loss, train_accuracy = nn.model.evaluate(X_train, y_train)
    # print('—> Training accuracy:', train_accuracy)

    # # Evaluate the model
    # loss, accuracy = nn.model.evaluate(X_test, y_test)
    # print('—> Test accuracy:', accuracy)

    # arr = nn.nn_to_array()
    # nn2 = array_to_nn(arr)

    # # Train the model
    # nn2.model.fit(X_train, y_train, epochs=100, batch_size=32)

    # # Evaluate the model on the training data
    # train_loss, train_accuracy = nn2.model.evaluate(X_train, y_train)
    # print('—> Training accuracy:', train_accuracy)

    # # Evaluate the model
    # loss, accuracy = nn2.model.evaluate(X_test, y_test)
    # print('—> Test accuracy:', accuracy)

    # # Genetic Algorithm using PyGAD
    # initial_population = [serialize({'activations': ['relu', 'sigmoid'], 'hidden_layer_sizes': [4, 32, 3]}) for _ in range(10)]

    # ga_instance = pygad.GA(
    #     initial_population=initial_population,
    #     fitness_func=fitness_func,
    #     num_generations=10,
    #     num_parents_mating=2,
    #     sol_per_pop=10,
    #     num_genes=len(initial_population[0]),
    #     crossover_type=custom_crossover,
    #     mutation_type=custom_mutation
    # )

    # ga_instance.run()

    # solution, solution_fitness, solution_idx = ga_instance.best_solution()
    # best_solution = deserialize(solution)

    # # Create a new NeuralNetwork instance using array_to_nn
    # nn2 = array_to_nn(best_solution)

    # # Now you can use nn2 for further training or evaluation
    # # For example, training and evaluating nn2 on your dataset
    # nn2.model.fit(X_train, y_train, epochs=100, batch_size=32)

    # # Evaluate the model on the training data
    # train_loss, train_accuracy = nn2.model.evaluate(X_train, y_train)
    # print('—> Training accuracy:', train_accuracy)

    # # Evaluate the model on the test data
    # test_loss, test_accuracy = nn2.model.evaluate(X_test, y_test)
    # print('—> Test accuracy:', test_accuracy)
    pass