import pprint
from torch import nn, optim
import sys
import gc
from matplotlib import pyplot as plt
import torch
sys.displayhook = pprint.pprint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEG_PARAMS = ("learning_rate", "batch_size", "epochs", "patience")

DATASET_LIST_CLASS = {
    # Classification Datasets
    'iris': 53,                     # Small, 4 features
    'wine': 109,                    # Small, 13 features
    'adult': 2,                     # Large, 14 features
    'breast_cancer_wisconsin_original': 15,  # Small, 9 features
    'heart_disease': 45,            # Medium, 13 features (mixed)
    'bank_marketing': 222,          # Medium, 16 features (mixed)
    'dry_bean': 602,                # Large, 16 features
    'optical_recognition_of_handwritten_digits': 80,  # Large, 64 features (image data)
}

DATASET_LIST_REG = {
    # Regression Datasets
    'auto_mpg': 9,                  # Small, 7 features (mixed)
    'energy_efficiency': 242,       # Small, 8 features
    'concrete_compressive_strength': 165,  # Medium, 8 features
    'abalone': 1,                   # Medium, 8 features
    'parkinsons_telemonitoring': 189,  # Medium, 26 features (time series data)
    'air_quality': 360,             # Medium, 15 features (time series data)
    'bike_sharing': 275,            # Medium, 16 features (mixed)
    'individual_household_electric_power_consumption': 235,  # Large, 7 features (time series data)
}






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
        elif activation_output == 'relu':
            self.output_activation = nn.ReLU()

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
                
                if problem_type == "classification":
                    batch_labels = batch_labels.long()
                elif problem_type == "regression":
                    batch_labels = batch_labels.view_as(outputs)
                
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            if val_loader:
                val_loss, val_accuracy = self.evaluate(val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    # print(f"Early stopping at epoch {epoch+1}")
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
                
                if problem_type == "classification":
                    # Ensure labels are of type Long for CrossEntropyLoss
                    batch_labels = batch_labels.long()
                    loss = self.criterion(outputs, batch_labels)
                    total_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                elif problem_type == "regression":
                    # Ensure the shapes match for MSELoss
                    batch_labels = batch_labels.view_as(outputs)
                    loss = self.criterion(outputs, batch_labels)
                    total_loss += loss.item()

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
            'INPUT_LAYER_SIZE': 4,
            'OUTPUT_LAYER_SIZE': 3
        },
        'wine': {
            'problem_type': "classification",
            'INPUT_LAYER_SIZE': 13,
            'OUTPUT_LAYER_SIZE': 3
        },
        'adult': {
            'problem_type': "classification",
            'INPUT_LAYER_SIZE': 14,
            'OUTPUT_LAYER_SIZE': 2
        },
        'breast_cancer_wisconsin_original': {
            'problem_type': "classification",
            'INPUT_LAYER_SIZE': 9,
            'OUTPUT_LAYER_SIZE': 2
        },
        'heart_disease': {
            'problem_type': "classification",
            'INPUT_LAYER_SIZE': 13,
            'OUTPUT_LAYER_SIZE': 2
        },
        'bank_marketing': {
            'problem_type': "classification",
            'INPUT_LAYER_SIZE': 16,
            'OUTPUT_LAYER_SIZE': 2
        },
        'dry_bean': {
            'problem_type': "classification",
            'INPUT_LAYER_SIZE': 16,
            'OUTPUT_LAYER_SIZE': 7
        },
        'optical_recognition_of_handwritten_digits': {
            'problem_type': "classification",
            'INPUT_LAYER_SIZE': 64,
            'OUTPUT_LAYER_SIZE': 10
        },
        'auto_mpg': {
            'problem_type': "regression",
            'INPUT_LAYER_SIZE': 7,
            'OUTPUT_LAYER_SIZE': 1
        },
        'energy_efficiency': {
            'problem_type': "regression",
            'INPUT_LAYER_SIZE': 8,
            'OUTPUT_LAYER_SIZE': 1
        },
        'concrete_compressive_strength': {
            'problem_type': "regression",
            'INPUT_LAYER_SIZE': 8,
            'OUTPUT_LAYER_SIZE': 1
        },
        'abalone': {
            'problem_type': "regression",
            'INPUT_LAYER_SIZE': 8,
            'OUTPUT_LAYER_SIZE': 1
        },
        'parkinsons_telemonitoring': {
            'problem_type': "regression",
            'INPUT_LAYER_SIZE': 26,
            'OUTPUT_LAYER_SIZE': 1
        },
        'air_quality': {
            'problem_type': "regression",
            'INPUT_LAYER_SIZE': 15,
            'OUTPUT_LAYER_SIZE': 1
        },
        'bike_sharing': {
            'problem_type': "regression",
            'INPUT_LAYER_SIZE': 16,
            'OUTPUT_LAYER_SIZE': 1
        },
        'individual_household_electric_power_consumption': {
            'problem_type': "regression",
            'INPUT_LAYER_SIZE': 7,
            'OUTPUT_LAYER_SIZE': 1
        }
    }


    if dataset not in dataset_parameters:
        raise ValueError(f"Dataset '{dataset}' is not defined.")

    # Extract the parameters for the dataset
    params = dataset_parameters[dataset]
    problem_type = params['problem_type']
    INPUT_LAYER_SIZE = params['INPUT_LAYER_SIZE']
    OUTPUT_LAYER_SIZE = params['OUTPUT_LAYER_SIZE']
    MAX_LAYERS = 20
    MAX_LAYER_SIZE = 512

    # Define activation functions based on the problem type
    ACTIVATIONS = {1: 'relu', 2: 'sigmoid', 3: 'tanh', 4: 'linear'}
    if problem_type == "classification":
        ACTIVATIONS_OUTPUT = {1: 'softmax', 2: 'sigmoid'}
    elif problem_type == "regression":
        ACTIVATIONS_OUTPUT = {1: 'relu', 2: 'sigmoid', 3: 'tanh', 4: 'linear'}

    return problem_type, MAX_LAYERS, MAX_LAYER_SIZE, INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, ACTIVATIONS, ACTIVATIONS_OUTPUT


# def plot_data(dataset):
#     _, ax = plt.subplots()
#     scatter = ax.scatter(dataset.data[:, 0], dataset.data[:, 1], c=dataset.target)
#     ax.set(xlabel=dataset.feature_names[0], ylabel=dataset.feature_names[1])
#     _ = ax.legend(scatter.legend_elements()[0], dataset.target_names, loc="lower right", title="Classes")
#     plt.show()
