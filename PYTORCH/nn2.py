import pprint
from torch import nn, optim
import sys
import gc
from matplotlib import pyplot as plt
import torch
sys.displayhook = pprint.pprint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEG_PARAMS = ("learning_rate", "batch_size", "epochs", "patience")

# Dataset selection and parameters
DATASET_LIST = [
    'iris',             # Classification
    'mnist',            # Classification
    'adult',            # Classification
    'wine',             # Classification
    'breast_cancer',    # Classification
    'heart_disease',    # Classification
    'thyroid_disease',  # Classification
    'census_income',    # Classification
    'california',       # Regression
    'diabetes',         # Regression
    'auto_mpg',         # Regression
    'concrete',         # Regression
    'abalone',          # Regression
    'housing',          # Regression
    'energy_efficiency',# Regression
    'kin8nm'            # Regression
]


dataset, problem_type, MAX_LAYERS, MAX_LAYER_SIZE, INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, ACTIVATIONS, ACTIVATIONS_OUTPUT = None, None, None, None, None, None, None, None

class NeuralNetwork(nn.Module):
    def __init__(self, learning_rate, batch_size, epochs, patience, num_layers, hidden_layer_sizes, activations, dropout_rates, batch_norms, activation_output):
        super(NeuralNetwork, self).__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.num_layers = num_layers
        self.hidden_layer_sizes = hidden_layer_sizes

        layer_sizes = [INPUT_LAYER_SIZE] + list(hidden_layer_sizes) + [OUTPUT_LAYER_SIZE]
        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            if layer_sizes[i] == 0 or layer_sizes[i+1] == 0:
                raise ValueError(f"Layer size must not be zero: {layer_sizes[i]} -> {layer_sizes[i+1]}")
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            if i < num_layers:  # Only apply to hidden layers
                if batch_norms[i]:
                    self.layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
                if activations[i] == 'relu':
                    self.layers.append(nn.ReLU())
                elif activations[i] == 'sigmoid':
                    self.layers.append(nn.Sigmoid())
                elif activations[i] == 'tanh':
                    self.layers.append(nn.Tanh())
                elif activations[i] == 'linear':
                    self.layers.append(nn.Identity())
                if dropout_rates[i] > 0:
                    self.layers.append(nn.Dropout(dropout_rates[i]))
        
        # Output activation
        if activation_output == 'softmax':
            self.output_activation = nn.Softmax(dim=1)
        elif activation_output == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif activation_output == 'tanh':
            self.output_activation = nn.Tanh()
        elif activation_output == 'linear':
            self.output_activation = nn.Identity()

        # Define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Define the loss function
        self.criterion = nn.CrossEntropyLoss() if problem_type == "classification" else nn.MSELoss()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        x = self.layers[-1](x)
        return self.output_activation(x)
    
    def train_model(self, train_loader, val_loader=None):
        self.train()  # Set the model to training mode
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            running_loss = 0.0
            for batch_data, batch_labels in train_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)  # Move data to GPU
                self.optimizer.zero_grad()
                outputs = self(batch_data)
                
                # Ensure labels are of type Long for CrossEntropyLoss
                if problem_type == "classification":
                    batch_labels = batch_labels.long()
                
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            if val_loader:
                val_loss, val_accuracy = self.evaluate(val_loader)
                # You can print or log the validation loss and accuracy here if needed
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    def evaluate(self, data_loader):
        self.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_data, batch_labels in data_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)  # Move data to GPU
                outputs = self(batch_data)
                
                # Ensure labels are of type Long for CrossEntropyLoss
                if problem_type == "classification":
                    batch_labels = batch_labels.long()
                
                loss = self.criterion(outputs, batch_labels)
                total_loss += loss.item()
                
                if problem_type == "classification":
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
        
        average_loss = total_loss / len(data_loader)
        accuracy = (correct / total) * 100 if problem_type == "classification" else None
        return average_loss, accuracy

def array_to_nn(ga_array):
    # print("Converting array to Neural Network")
    # print(ga_array)
    learning_rate = float(ga_array[0])
    batch_size = int(ga_array[1])
    epochs = int(ga_array[2])
    patience = int(ga_array[3])
    num_layers = int(ga_array[len(BEG_PARAMS)])
    hidden_layer_sizes = ga_array[len(BEG_PARAMS) + 1:num_layers + len(BEG_PARAMS) + 1]
    activations_keys = ga_array[MAX_LAYERS + len(BEG_PARAMS) + 1:num_layers + MAX_LAYERS + len(BEG_PARAMS) + 1]
    dropout_rates = ga_array[2 * MAX_LAYERS + len(BEG_PARAMS) + 1:num_layers + 2 * MAX_LAYERS + len(BEG_PARAMS) + 1]
    batch_norms = ga_array[3 * MAX_LAYERS + len(BEG_PARAMS) + 1:num_layers + 3 * MAX_LAYERS + len(BEG_PARAMS) + 1]
    activation_output_key = int(ga_array[-1])

    activations = [ACTIVATIONS[int(key)] if int(key) != -1 else 'linear' for key in activations_keys]
    activation_output = ACTIVATIONS_OUTPUT[activation_output_key]

    # print("Learning Rate:", learning_rate)
    # print("Batch Size:", batch_size)
    # print("Epochs:", epochs)
    # print("Patience:", patience)
    # print("Number of Layers:", num_layers)
    # print("Hidden Layer Sizes:", hidden_layer_sizes)
    # print("Activations:", activations)
    # print("Dropout Rates:", dropout_rates)
    # print("Batch Norms:", batch_norms)
    # print("Output Activation:", activation_output)
    
    nn = NeuralNetwork(learning_rate=learning_rate,
                       batch_size=batch_size,
                       epochs=epochs,
                       patience=patience,
                       num_layers=num_layers,
                       hidden_layer_sizes=list(map(int, hidden_layer_sizes)),
                       activations=activations,
                       dropout_rates=list(map(float, dropout_rates)),
                       batch_norms=list(map(int, batch_norms)),
                       activation_output=activation_output).to(device) 


    return nn

def load_and_define_parameters(dataset):
    global problem_type, MAX_LAYERS, MAX_LAYER_SIZE, INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, ACTIVATIONS, ACTIVATIONS_OUTPUT

    dataset_parameters = {
        'iris': {
            'problem_type': "classification",
            'MAX_LAYERS': 10,
            'MAX_LAYER_SIZE': 24,
            'INPUT_LAYER_SIZE': 4,
            'OUTPUT_LAYER_SIZE': 3
        },
        'mnist': {
            'problem_type': "classification",
            'MAX_LAYERS': 10,
            'MAX_LAYER_SIZE': 512,
            'INPUT_LAYER_SIZE': 784,
            'OUTPUT_LAYER_SIZE': 10
        },
        'adult': {
            'problem_type': "classification",
            'MAX_LAYERS': 10,
            'MAX_LAYER_SIZE': 128,
            'INPUT_LAYER_SIZE': 108,
            'OUTPUT_LAYER_SIZE': 2
        },
        'wine': {
            'problem_type': "classification",
            'MAX_LAYERS': 10,
            'MAX_LAYER_SIZE': 24,
            'INPUT_LAYER_SIZE': 13,
            'OUTPUT_LAYER_SIZE': 3
        },
        'breast_cancer': {
            'problem_type': "classification",
            'MAX_LAYERS': 10,
            'MAX_LAYER_SIZE': 32,
            'INPUT_LAYER_SIZE': 33,
            'OUTPUT_LAYER_SIZE': 2
        },
        'heart_disease': {
            'problem_type': "classification",
            'MAX_LAYERS': 10,
            'MAX_LAYER_SIZE': 32,
            'INPUT_LAYER_SIZE': 13,
            'OUTPUT_LAYER_SIZE': 2
        },
        'thyroid_disease': {
            'problem_type': "classification",
            'MAX_LAYERS': 10,
            'MAX_LAYER_SIZE': 64,
            'INPUT_LAYER_SIZE': 54,
            'OUTPUT_LAYER_SIZE': 3
        },
        'census_income': {
            'problem_type': "classification",
            'MAX_LAYERS': 10,
            'MAX_LAYER_SIZE': 256,
            'INPUT_LAYER_SIZE': 509,
            'OUTPUT_LAYER_SIZE': 2
        },
        'california': {
            'problem_type': "regression",
            'MAX_LAYERS': 10,
            'MAX_LAYER_SIZE': 64,
            'INPUT_LAYER_SIZE': 8,
            'OUTPUT_LAYER_SIZE': 1
        },
        'diabetes': {
            'problem_type': "regression",
            'MAX_LAYERS': 10,
            'MAX_LAYER_SIZE': 32,
            'INPUT_LAYER_SIZE': 10,
            'OUTPUT_LAYER_SIZE': 1
        },
        'auto_mpg': {
            'problem_type': "regression",
            'MAX_LAYERS': 10,
            'MAX_LAYER_SIZE': 32,
            'INPUT_LAYER_SIZE': 7,
            'OUTPUT_LAYER_SIZE': 1
        },
        'concrete': {
            'problem_type': "regression",
            'MAX_LAYERS': 10,
            'MAX_LAYER_SIZE': 32,
            'INPUT_LAYER_SIZE': 8,
            'OUTPUT_LAYER_SIZE': 1
        },
        'abalone': {
            'problem_type': "regression",
            'MAX_LAYERS': 10,
            'MAX_LAYER_SIZE': 32,
            'INPUT_LAYER_SIZE': 8,
            'OUTPUT_LAYER_SIZE': 1
        },
        'housing': {
            'problem_type': "regression",
            'MAX_LAYERS': 10,
            'MAX_LAYER_SIZE': 32,
            'INPUT_LAYER_SIZE': 13,
            'OUTPUT_LAYER_SIZE': 1
        },
        'energy_efficiency': {
            'problem_type': "regression",
            'MAX_LAYERS': 10,
            'MAX_LAYER_SIZE': 32,
            'INPUT_LAYER_SIZE': 8,
            'OUTPUT_LAYER_SIZE': 1
        },
        'kin8nm': {
            'problem_type': "regression",
            'MAX_LAYERS': 10,
            'MAX_LAYER_SIZE': 64,
            'INPUT_LAYER_SIZE': 8,
            'OUTPUT_LAYER_SIZE': 1
        }
    }

    if dataset not in dataset_parameters:
        raise ValueError(f"Dataset '{dataset}' is not defined.")

    # Extract the parameters for the dataset
    params = dataset_parameters[dataset]
    problem_type = params['problem_type']
    MAX_LAYERS = params['MAX_LAYERS']
    MAX_LAYER_SIZE = params['MAX_LAYER_SIZE']
    INPUT_LAYER_SIZE = params['INPUT_LAYER_SIZE']
    OUTPUT_LAYER_SIZE = params['OUTPUT_LAYER_SIZE']

    # Define activation functions based on the problem type
    if problem_type == "classification":
        ACTIVATIONS = {1: 'relu', 2: 'sigmoid', 3: 'tanh', 4: 'linear'}
        ACTIVATIONS_OUTPUT = {1: 'softmax', 2: 'sigmoid'}
    elif problem_type == "regression":
        ACTIVATIONS = {1: 'relu', 2: 'sigmoid', 3: 'tanh', 4: 'linear'}
        ACTIVATIONS_OUTPUT = {1: 'linear', 2: 'relu', 3: 'sigmoid', 4: 'tanh'}

    return problem_type, MAX_LAYERS, MAX_LAYER_SIZE, INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, ACTIVATIONS, ACTIVATIONS_OUTPUT


# def plot_data(dataset):
#     _, ax = plt.subplots()
#     scatter = ax.scatter(dataset.data[:, 0], dataset.data[:, 1], c=dataset.target)
#     ax.set(xlabel=dataset.feature_names[0], ylabel=dataset.feature_names[1])
#     _ = ax.legend(scatter.legend_elements()[0], dataset.target_names, loc="lower right", title="Classes")
#     plt.show()
