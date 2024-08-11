import sys
import pprint
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

sys.displayhook = pprint.pprint

# dataset = 'iris'
dataset = 'mnist'
# dataset = 'california'
# dataset = 'diabetes'

BEG_PARAMS = ("learning_rate", "batch_size", "epochs", "patience") 

# Set global parameters based on the dataset
if dataset == 'iris':
    problem_type = "classification"
    MAX_LAYERS =  6
    MAX_LAYER_SIZE = 256
    INPUT_LAYER_SIZE = 4  
    OUTPUT_LAYER_SIZE = 3
elif dataset == 'mnist':
    problem_type = "classification"
    MAX_LAYERS = 6
    MAX_LAYER_SIZE = 128
    INPUT_LAYER_SIZE = 28 * 28
    OUTPUT_LAYER_SIZE = 10
elif dataset == 'california':
    problem_type = "regression"
    MAX_LAYERS = 6
    MAX_LAYER_SIZE = 256
    INPUT_LAYER_SIZE = 8
    OUTPUT_LAYER_SIZE = 1
elif dataset == 'diabetes':
    problem_type = "regression"
    MAX_LAYERS = 6
    MAX_LAYER_SIZE = 256
    INPUT_LAYER_SIZE = 10
    OUTPUT_LAYER_SIZE = 1
    
if problem_type == "classification":
    ACTIVATIONS = {1: 'relu', 2: 'sigmoid', 3: 'tanh'}
    ACTIVATIONS_OUTPUT = {1: 'softmax'}
elif problem_type == "regression":
    ACTIVATIONS = {1: 'relu', 2: 'sigmoid', 3: 'tanh', 4: 'linear'}
    ACTIVATIONS_OUTPUT = {1: 'linear'}
        
class NeuralNetwork():
    def __init__(self, learning_rate, batch_size, epochs, patience, num_layers, hidden_layer_sizes, activations, dropout_rates, batch_norms, activation_output):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.num_layers = num_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activations = activations
        self.dropout_rates = dropout_rates
        self.batch_norms = batch_norms
        self.activation_output = activation_output
        self.model = None
        self.early_stopping = None
        
        self.create_model()
        
    #@tf.function
    def create_model(self):
        # Define the input layer
        input_layer = Input(shape=(INPUT_LAYER_SIZE,), name='input')

        hidden_layer = input_layer

        # Define the hidden layers
        for i in range(0, self.num_layers):
            hidden_layer_size = int(self.hidden_layer_sizes[i])  # Ensure hidden_layer_sizes[i] is an integer

            if i == 0:
                hidden_layer = Dense(hidden_layer_size, name='hidden_' + str(i + 1))(input_layer)
            else:
                hidden_layer = Dense(hidden_layer_size, name='hidden_' + str(i + 1))(hidden_layer)

            if tf.not_equal(self.batch_norms[i], -1):
                hidden_layer = BatchNormalization()(hidden_layer)

            hidden_layer = Activation(self.activations[i])(hidden_layer)

            if tf.reduce_any(tf.math.greater(self.dropout_rates[i], 0)):
                hidden_layer = Dropout(self.dropout_rates[i])(hidden_layer)

        # Define the output layer
        output_layer = Dense(OUTPUT_LAYER_SIZE, name='output')(hidden_layer)
        if tf.not_equal(self.batch_norms[-1], -1):
            output_layer = BatchNormalization()(output_layer)
        output_layer = Activation(self.activation_output)(output_layer)

        # Create the model
        self.model = Model(inputs=input_layer, outputs=output_layer)

        # Set the learning rate
        optimizer = Adam(learning_rate=float(self.learning_rate))  # Ensure learning rate is a float
        
        if problem_type == "classification":
            self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        elif problem_type == "regression":
            self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
        
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)


def array_to_nn(ga_array):
    # Extract values directly from the tensor without converting to NumPy
    learning_rate = ga_array[0]
    batch_size = tf.cast(ga_array[1], tf.int32)
    epochs = tf.cast(ga_array[2], tf.int32)
    patience = tf.cast(ga_array[3], tf.int32)
    num_layers = tf.cast(ga_array[len(BEG_PARAMS)], tf.int32)

    # Use the tensor slices directly
    hidden_layer_sizes = tf.cast(ga_array[len(BEG_PARAMS) + 1:MAX_LAYERS + len(BEG_PARAMS) + 1], tf.int32)
    activations_keys = tf.cast(ga_array[MAX_LAYERS + len(BEG_PARAMS) + 1:2 * MAX_LAYERS + len(BEG_PARAMS) + 1], tf.int32)
    dropout_rates = tf.cast(ga_array[2 * MAX_LAYERS + len(BEG_PARAMS) + 1:3 * MAX_LAYERS + len(BEG_PARAMS) + 1], tf.float32)
    batch_norms = tf.cast(ga_array[3 * MAX_LAYERS + len(BEG_PARAMS) + 1:4 * MAX_LAYERS + len(BEG_PARAMS) + 1], tf.int32)
    activation_output_key = tf.cast(ga_array[4 * MAX_LAYERS + len(BEG_PARAMS) + 1], tf.int32)

    # Convert activations_keys to corresponding activation functions
    activations = [ACTIVATIONS[key] if key != -1 else key for key in activations_keys.numpy()]

    # Convert the output activation key to the actual activation function
    activation_output = ACTIVATIONS_OUTPUT[int(activation_output_key)] if activation_output_key != -1 else activation_output_key

    nn = NeuralNetwork(
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        num_layers=num_layers,
        hidden_layer_sizes=hidden_layer_sizes,
        activations=activations,
        dropout_rates=dropout_rates,
        batch_norms=batch_norms,
        activation_output=activation_output
    )

    return nn



def plot_data(dataset):
    _, ax = plt.subplots()
    scatter = ax.scatter(dataset.data[:, 0], dataset.data[:, 1], c=dataset.target)
    ax.set(xlabel=dataset.feature_names[0], ylabel=dataset.feature_names[1])
    _ = ax.legend(scatter.legend_elements()[0], dataset.target_names, loc="lower right", title="Classes")
    plt.show()