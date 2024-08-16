import logging
import sys
import os
import gc
import time
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np
import psutil
import pygad
from tqdm import tqdm
import pandas as pd
import torch
import enlighten

from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris, fetch_california_housing, load_diabetes, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nn2 import *

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    print("CUDA is available! PyTorch is using the GPU.")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. PyTorch is using the CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

manager = enlighten.get_manager()

# tracemalloc.start()

test_size = 0.3
min_delta = 0.001  # Minimum change in fitness to qualify as an improvement
patience_ga = 30  # Number of generations to wait before stopping if there is no improvement
penalty_mult_list = [0, 0.01, 0.05, 0.1, 0.5, 1, 2, 5]  # Penalty multiplier for the complexity of the network

# fitness_scores = {}
fitness_history_best = []
fitness_history_avg = []
best_fitness = -np.inf
patience_counter = 0
generation_counter = 1

gen_num_printed = True

# Ensure the directory exists
log_dir = f"./logs/{problem_type}/{dataset}"
os.makedirs(log_dir, exist_ok=True)

# Set up logging
# logging.basicConfig(filename=f"{log_dir}/memory_usage.log", level=logging.INFO, format='%(asctime)s - %(message)s')

# def log_memory_usage(message="Memory usage"):
#     process = psutil.Process()
#     memory_info = process.memory_info()
#     logging.info(f"{message}: RSS={memory_info.rss / (1024 ** 2):.2f} MB, VMS={memory_info.vms / (1024 ** 2):.2f} MB")
    
#     # # Also log the tracemalloc stats
#     # snapshot = tracemalloc.take_snapshot()
#     # top_stats = snapshot.statistics('lineno')
#     # logging.info(f"[ Top 10 memory-consuming lines for {message} ]")
#     # for stat in top_stats[:10]:
#     #     logging.info(stat)
        
        
"""
Datasets Included

Classification:
Iris - Small, 4 features.
MNIST - Large, image data with 784 features.
Adult - Medium, 14 features.
Wine - Small, 13 features.
Breast Cancer - Medium, mixed features.
Heart Disease - Medium, mixed features.
Thyroid Disease - Medium, mixed features.
Census Income - Large, demographic features.

Regression:
California Housing - Large, 8 features.
Diabetes - Medium, 10 features.
Auto MPG - Small, mixed features.
Concrete - Medium, 8 features.
Abalone - Medium, 8 features.
Housing - Small, 14 features.
Energy Efficiency - Small, mixed features.
Kin8nm - Medium, mixed features."""


def load_and_preprocess_data(dataset):
    if dataset == 'iris':
        iris = load_iris()
        X = iris.data.astype("float32")
        y = iris.target.astype("int64")

    elif dataset == 'mnist':
        mnist = fetch_openml('mnist_784', version=1)
        X, y = mnist["data"].astype("float32") / 255.0, mnist["target"].astype("int64")

    elif dataset == 'california':
        california = fetch_california_housing()
        X = california.data.astype("float32")
        y = california.target.astype("float32")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    elif dataset == 'diabetes':
        diabetes = load_diabetes()
        X = diabetes.data.astype("float32")
        y = diabetes.target.astype("float32")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Below are the additional datasets from UCI repository
    
    elif dataset == 'adult':
        data = fetch_openml('adult', version=1)
        X, y = pd.get_dummies(data.data), LabelEncoder().fit_transform(data.target)
        X = X.astype("float32")

    elif dataset == 'wine':
        data = fetch_openml('wine', version=1)
        X, y = data.data, LabelEncoder().fit_transform(data.target)
        X = X.astype("float32")

    elif dataset == 'breast_cancer':
        data = fetch_openml('breast-cancer', version=1)
        X, y = pd.get_dummies(data.data), LabelEncoder().fit_transform(data.target)
        X = X.astype("float32")

    elif dataset == 'heart_disease':
        data = fetch_openml('heart-disease', version=1)
        X, y = pd.get_dummies(data.data), LabelEncoder().fit_transform(data.target)
        X = X.astype("float32")

    elif dataset == 'thyroid_disease':
        data = fetch_openml('thyroid-disease', version=1)
        X, y = pd.get_dummies(data.data), LabelEncoder().fit_transform(data.target)
        X = X.astype("float32")

    elif dataset == 'census_income':
        data = fetch_openml('census-income', version=1)
        X, y = pd.get_dummies(data.data), LabelEncoder().fit_transform(data.target)
        X = X.astype("float32")

    elif dataset == 'auto_mpg':
        data = fetch_openml('autoMpg', version=1)
        X, y = pd.get_dummies(data.data), data.target
        X = X.astype("float32")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    elif dataset == 'concrete':
        data = fetch_openml('concrete', version=1)
        X, y = data.data, data.target
        X = X.astype("float32")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    elif dataset == 'abalone':
        data = fetch_openml('abalone', version=1)
        X, y = pd.get_dummies(data.data), data.target
        X = X.astype("float32")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    elif dataset == 'housing':
        data = fetch_openml('housing', version=1)
        X, y = pd.get_dummies(data.data), data.target
        X = X.astype("float32")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    elif dataset == 'energy_efficiency':
        data = fetch_openml('energy-efficiency', version=1)
        X, y = pd.get_dummies(data.data), data.target
        X = X.astype("float32")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    elif dataset == 'kin8nm':
        data = fetch_openml('kin8nm', version=1)
        X, y = pd.get_dummies(data.data), data.target
        X = X.astype("float32")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Convert NumPy arrays to PyTorch tensors
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def callback_generation(ga_instance):
    global best_fitness, patience_counter, min_delta, patience_ga, generation_counter, gen_num_printed, fitness_scores
    
    # Save the fitness score for the best and the average solution in each generation
    best_fitness_current = np.max(ga_instance.last_generation_fitness)
    fitness_history_best.append(best_fitness_current)
    fitness_history_avg.append(np.mean(ga_instance.last_generation_fitness))

    # Early stopping logic
    if best_fitness_current - best_fitness > min_delta:
        patience_counter = 0
        best_fitness = best_fitness_current
    else:
        patience_counter += 1
    
    # Log memory usage after each generation
    # log_memory_usage(f"After generation {ga_instance.generations_completed + 1}")
    # Early stopping check
    if patience_counter >= patience_ga:
        # print(f"\nEarly stopping: no improvement in fitness for {patience_ga} generations.\n")
        ticks_generation.count = ticks_generation.total
        ticks_generation.refresh()
        return "stop"
    
    ticks_generation.update()
    # print(f"\n—————————— GENERATION {ga_instance.generations_completed + 1} ——————————\n")

def generatePopulation(sol_per_pop):
    population = []
    
    for _ in range(sol_per_pop):
        # Generate random values for the parameters
        learning_rate = np.random.uniform(0.0001, 0.01)
        batch_size = np.random.choice([16, 32, 64, 128, 256])
        epochs = np.random.randint(1, 50)
        patience = np.random.randint(1, 10)
        num_layers = np.random.randint(1, MAX_LAYERS + 1)
        
        hidden_layer_sizes = np.random.randint(1, MAX_LAYER_SIZE + 1, size=num_layers)
        activations = np.random.randint(1, len(ACTIVATIONS) + 1, size=num_layers)
        dropout_rates = np.random.uniform(0.0, 0.5, size=num_layers)
        batch_norms = np.random.randint(0, 2, size=num_layers)  # 0 or 1 for batch normalization
        activation_output = np.random.randint(1, len(ACTIVATIONS_OUTPUT) + 1)
        
        # Pad the arrays to MAX_LAYERS length
        hidden_layer_sizes = np.append(hidden_layer_sizes, [0] * (MAX_LAYERS - num_layers))
        activations = np.append(activations, [-1] * (MAX_LAYERS - num_layers))
        dropout_rates = np.append(dropout_rates, [-1.0] * (MAX_LAYERS - num_layers))
        batch_norms = np.append(batch_norms, [-1] * (MAX_LAYERS - num_layers))
        
        # Combine all parts into a single solution array
        solution = np.concatenate((
            [learning_rate, batch_size, epochs, patience, num_layers],
            hidden_layer_sizes.astype(np.float32),
            activations.astype(np.float32),
            dropout_rates.astype(np.float32),
            batch_norms.astype(np.float32),
            [activation_output]
        ))
        
        # Append the solution to the population
        population.append(solution)
        
        # # Free memory after each iteration
        # del hidden_layer_sizes, activations, dropout_rates, batch_norms, solution
        # gc.collect()
    
    # Convert the list of solutions to a numpy array
    population = np.array(population)
    # print("\n——————————— GENERATION 0 ———————————\n")
    
    return population


def fitness_func(ga_instance, solution, solution_idx):
    global gen_num_printed
    # print(".. Fitness function ..")    

    # Create a neural network from the solution array
    solution_nn = array_to_nn(solution).to(device)
    
    # Create DataLoader for training and validation
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=int(solution_nn.batch_size), shuffle=True)
    val_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=int(solution_nn.batch_size), shuffle=False)

    # Train the neural network
    solution_nn.train_model(train_loader, val_loader)

    # Evaluate the model on the validation set
    validation_loss, validation_accuracy = solution_nn.evaluate(val_loader)

    # Check if validation_loss is NaN
    if torch.isnan(torch.tensor(validation_loss)):
        # print("Validation loss is NaN, setting fitness score to a very low value.")
        fitness_score = -np.inf
    else:
        # Calculate the number of layers and total number of neurons
        num_layers = solution_nn.num_layers
        total_neurons = sum(solution_nn.hidden_layer_sizes)
        
        # Calculate the relative complexity of the network
        relative_layers = num_layers / MAX_LAYERS
        relative_neurons = total_neurons / (MAX_LAYERS * MAX_LAYER_SIZE)  # Maximum possible neurons
        
        layer_mult = 0.1
        neuron_mult = 0.01
        
        # Calculate the penalty based on relative complexity
        penalty = relative_layers * layer_mult + relative_neurons * neuron_mult
        
        # Calculate the fitness score
        small_value = 1e-8
        fitness_score = 1 / (validation_loss + penalty_mult * penalty + small_value)
        # print("Fitness score: ", fitness_score)

    return fitness_score


def custom_crossover(parents, offspring_size, ga_instance):
    # print("... Crossover ...")
    # print("—> offspring size: ", offspring_size[0])
    
    offspring = np.array([])
    for i in range(offspring_size[0]):
        # print(f"\n{i}:\n")
        parent1_idx = np.random.randint(0, len(parents))
        parent2_idx = np.random.randint(0, len(parents))
        parent1 = parents[parent1_idx]
        parent2 = parents[parent2_idx]
        offspring1, offspring2 = structured_crossover(parent1, parent2)
        if offspring.size == 0:
            offspring = np.array(offspring1)
        else:
            offspring = np.vstack((offspring, offspring1))
        offspring = np.vstack((offspring, offspring2))
    return np.array(offspring[:offspring_size[0]])


def structured_crossover(parent1, parent2):
    # Parameters of parent 1
    learning_rate1 = float(parent1[0])
    batch_size1 = int(parent1[1])
    epochs1 = int(parent1[2])
    patience1 = int(parent1[3])
    num_layers1 = int(parent1[len(BEG_PARAMS)])
    hidden_layer_sizes1 = parent1[len(BEG_PARAMS) + 1:num_layers1 + len(BEG_PARAMS) + 1].astype(np.int32)
    activations1 = parent1[MAX_LAYERS + len(BEG_PARAMS) + 1:MAX_LAYERS + num_layers1 + len(BEG_PARAMS) + 1].astype(np.int32)
    dropout_rates1 = parent1[2 * MAX_LAYERS + len(BEG_PARAMS) + 1:2 * MAX_LAYERS + num_layers1 + len(BEG_PARAMS) + 1]
    batch_norms1 = parent1[3 * MAX_LAYERS + len(BEG_PARAMS) + 1:3 * MAX_LAYERS + num_layers1 + len(BEG_PARAMS) + 1].astype(np.int32)
    
    # Parameters of parent 2
    learning_rate2 = float(parent2[0])
    batch_size2 = int(parent2[1])
    epochs2 = int(parent2[2])
    patience2 = int(parent2[3])
    num_layers2 = int(parent2[len(BEG_PARAMS)])
    hidden_layer_sizes2 = parent2[len(BEG_PARAMS) + 1:num_layers2 + len(BEG_PARAMS) + 1].astype(np.int32)
    activations2 = parent2[MAX_LAYERS + len(BEG_PARAMS) + 1:MAX_LAYERS + num_layers2 + len(BEG_PARAMS) + 1].astype(np.int32)
    dropout_rates2 = parent2[2 * MAX_LAYERS + len(BEG_PARAMS) + 1:2 * MAX_LAYERS + num_layers2 + len(BEG_PARAMS) + 1]
    batch_norms2 = parent2[3 * MAX_LAYERS + len(BEG_PARAMS) + 1:3 * MAX_LAYERS + num_layers2 + len(BEG_PARAMS) + 1].astype(np.int32)
    
    ## CROSSOVER ##
    cross_option = np.random.randint(1, 3)
    
    # OPTION 1: beginning of parent 1 + "connection layer" + end of parent 2
    if cross_option == 1:
        # print("Option 1")
        # Select crossover points
        parent1_point1 = parent1_point2 = parent2_point1 = parent2_point2 = -1
        
        if num_layers1 == 1:
            parent1_point1 = 1
            parent1_point2 = 0
        else:
            parent1_point1 = np.random.randint(1, min(num_layers1, MAX_LAYERS - 2)) # -2 to save place for one connection layer and at least one ending layer from parent 2
        
        if num_layers2 == 1:
            parent2_point1 = 0
            parent2_point2 = 1
        else:
            parent2_point1 = np.random.randint(max(0, num_layers2 - (MAX_LAYERS - parent1_point1 - 1)), num_layers2)
            parent2_point2 = np.random.randint(1, min(num_layers2, MAX_LAYERS - 2))
        
        if parent1_point2 == -1:
            parent1_point2 = np.random.randint(max(0, num_layers1 - (MAX_LAYERS - parent2_point2 - 1)), num_layers1)

        connection_layer_size = np.random.randint(1, MAX_LAYER_SIZE + 1, (2,))
        connection_activation = np.random.randint(1, len(ACTIVATIONS) + 1, (2,))
        connection_dropout = np.random.uniform(0.1, 0.5, (2,))
        connection_batch_norm = np.random.choice([0, 1], (2,))
        
        # print(f"——> parent 1:")
        # print(f"    —> point 1: {parent1_point1}, point 2: {parent1_point2}")
        # print(f"——> parent 2:")
        # print(f"    —> point 1: {parent2_point1}, point 2: {parent2_point2}")
        
        # Perform crossover
        new_learning_rate1 = float(np.random.choice([learning_rate1, learning_rate2]))
        new_learning_rate2 = float(np.random.choice([learning_rate1, learning_rate2]))
        
        new_batch_size1 = float(np.random.choice([batch_size1, batch_size2]))
        new_batch_size2 = float(np.random.choice([batch_size1, batch_size2]))
        
        new_epochs1 = float(np.random.choice([epochs1, epochs2]))
        new_epochs2 = float(np.random.choice([epochs1, epochs2]))
        
        new_patience1 = float(np.random.choice([patience1, patience2]))
        new_patience2 = float(np.random.choice([patience1, patience2]))
        
        new_hidden_layer_sizes1 = np.concatenate((hidden_layer_sizes1[:parent1_point1], [connection_layer_size[0]], hidden_layer_sizes2[parent2_point1:]))
        new_hidden_layer_sizes2 = np.concatenate((hidden_layer_sizes2[:parent2_point2], [connection_layer_size[1]], hidden_layer_sizes1[parent1_point2:]))
        
        new_num_layers1 = len(new_hidden_layer_sizes1)
        new_num_layers2 = len(new_hidden_layer_sizes2)
        
        new_activations1 = np.concatenate((activations1[:parent1_point1], [connection_activation[0]], activations2[parent2_point1:]))
        new_activations2 = np.concatenate((activations2[:parent2_point2], [connection_activation[1]], activations1[parent1_point2:]))
        
        new_dropout_rates1 = np.concatenate((dropout_rates1[:parent1_point1], [connection_dropout[0]], dropout_rates2[parent2_point1:]))
        new_dropout_rates2 = np.concatenate((dropout_rates2[:parent2_point2], [connection_dropout[1]], dropout_rates1[parent1_point2:]))
        
        new_batch_norms1 = np.concatenate((batch_norms1[:parent1_point1], [connection_batch_norm[0]], batch_norms2[parent2_point1:]))
        new_batch_norms2 = np.concatenate((batch_norms2[:parent2_point2], [connection_batch_norm[1]], batch_norms1[parent1_point2:]))
    
        # Pad each part to MAX_LAYERS length
        new_hidden_layer_sizes1 = np.append(new_hidden_layer_sizes1, [0] * (MAX_LAYERS - new_num_layers1))  # 0 indicates padding
        new_hidden_layer_sizes2 = np.append(new_hidden_layer_sizes2, [0] * (MAX_LAYERS - new_num_layers2))
        new_activations1 = np.append(new_activations1, [-1] * (MAX_LAYERS - new_num_layers1))               # -1 indicates padding
        new_activations2 = np.append(new_activations2, [-1] * (MAX_LAYERS - new_num_layers2))
        new_dropout_rates1 = np.append(new_dropout_rates1, [-1.0] * (MAX_LAYERS - new_num_layers1))         # -1 indicates padding
        new_dropout_rates2 = np.append(new_dropout_rates2, [-1.0] * (MAX_LAYERS - new_num_layers2))
        new_batch_norms1 = np.append(new_batch_norms1, [-1] * (MAX_LAYERS - new_num_layers1))               # -1 indicates padding
        new_batch_norms2 = np.append(new_batch_norms2, [-1] * (MAX_LAYERS - new_num_layers2))
        
        offspring1 = np.concatenate((
            [new_learning_rate1, new_batch_size1, new_epochs1, new_patience1, new_num_layers1],
            new_hidden_layer_sizes1.astype(np.float32), 
            new_activations1.astype(np.float32), 
            new_dropout_rates1.astype(np.float32), 
            new_batch_norms1.astype(np.float32), 
            [parent2[-1]]
        ))
        
        offspring2 = np.concatenate((
            [new_learning_rate2, new_batch_size2, new_epochs2, new_patience2, new_num_layers2],
            new_hidden_layer_sizes2.astype(np.float32), 
            new_activations2.astype(np.float32), 
            new_dropout_rates2.astype(np.float32), 
            new_batch_norms2.astype(np.float32), 
            [parent1[-1]]
        ))
        
    # OPTION 2: beginning of parent 1 + "connection layer" + middle of parent 2 + "connection layer" + end of parent 1
    elif cross_option == 2:
        # print("Option 2")
        offspring1 = []
        offspring2 = []
        for i in range(2):
            if i == 1:
                # Parameters of parent 1
                learning_rate2 = float(parent1[0])
                batch_size2 = int(parent1[1])
                epochs2 = int(parent1[2])
                patience2 = int(parent1[3])
                num_layers2 = int(parent1[len(BEG_PARAMS)])
                hidden_layer_sizes2 = parent1[len(BEG_PARAMS) + 1:num_layers2 + len(BEG_PARAMS) + 1].astype(np.int32)
                activations2 = parent1[MAX_LAYERS + len(BEG_PARAMS) + 1:MAX_LAYERS + num_layers2 + len(BEG_PARAMS) + 1].astype(np.int32)
                dropout_rates2 = parent1[2 * MAX_LAYERS + len(BEG_PARAMS) + 1:2 * MAX_LAYERS + num_layers2 + len(BEG_PARAMS) + 1]
                batch_norms2 = parent1[3 * MAX_LAYERS + len(BEG_PARAMS) + 1:3 * MAX_LAYERS + num_layers2 + len(BEG_PARAMS) + 1].astype(np.int32)
                
                # Parameters of parent 2
                learning_rate1 = float(parent2[0])
                batch_size1 = int(parent2[1])
                epochs1 = int(parent2[2])
                patience1 = int(parent2[3])
                num_layers1 = int(parent2[len(BEG_PARAMS)])
                hidden_layer_sizes1 = parent2[len(BEG_PARAMS) + 1:num_layers1 + len(BEG_PARAMS) + 1].astype(np.int32)
                activations1 = parent2[MAX_LAYERS + len(BEG_PARAMS) + 1:MAX_LAYERS + num_layers1 + len(BEG_PARAMS) + 1].astype(np.int32)
                dropout_rates1 = parent2[2 * MAX_LAYERS + len(BEG_PARAMS) + 1:2 * MAX_LAYERS + num_layers1 + len(BEG_PARAMS) + 1]
                batch_norms1 = parent2[3 * MAX_LAYERS + len(BEG_PARAMS) + 1:3 * MAX_LAYERS + num_layers1 + len(BEG_PARAMS) + 1].astype(np.int32)
                
            # Generate connection layer parameters
            connection_layer_size = np.random.randint(1, MAX_LAYER_SIZE + 1, (2,))
            connection_activation = np.random.randint(1, len(ACTIVATIONS) + 1, (2,))
            connection_dropout = np.random.uniform(0.1, 0.5, (2,))
            connection_batch_norm = np.random.choice([0, 1], (2,))
            
            # Select crossover points
            if num_layers1 == 1:
                # If parent 1 has only one layer, we randomly choose whether we place the layer at the beginning or end
                parent1_position = 1 if np.random.rand() < 0.5 else 2 # 1 for beginning, 2 for end
                part1_size = 1
            else:
                parent1_position = 0 # 0 for default
                if num_layers1 == 2:
                    # If parent 1 has two layers, we place one layer at the beginning and the other at the end
                    parent1_point11 = parent1_point12 = 1
                    part1_size = 2
                else:
                    # If parent 1 has more than two layers, we randomly choose a number of beginning layers for the beginning and ending layers for the end
                    parent1_point11 = np.random.randint(1, min(num_layers1 - 1, MAX_LAYERS - 4)) # 4: 2 connection layers, 1 middle layer from parent 2 and 1 ending layer from parent 1
                    parent1_point12 = np.random.randint(max(parent1_point11, num_layers1 - (MAX_LAYERS - parent1_point11 - 3)), num_layers1)
                    part1_size = parent1_point11 + (num_layers1 - parent1_point12)
            
            part2_max_size = MAX_LAYERS - part1_size - 2
            # part2_size = np.random.randint(1, min(part2_max_size))
            if num_layers2 == 1:
                # If parent 2 has only one layer, we place the layer in the middle
                parent2_point11 = 0
                parent2_point12 = 1
            elif num_layers2 == 2:
                # If parent 2 has two layers, we randomly choose whether we place one or both layers in the middle
                part2_size = min(part2_max_size, np.random.choice([1, 2]))
                if part2_size == 1:
                    parent2_point11 = np.random.choice([0, 1])
                    parent2_point12 = parent2_point11 + 1
                else:
                    parent2_point11 = 0
                    parent2_point12 = 2
            else:
                # If parent 2 has more than two layers, we randomly select two points within the range of the number of layers minus first and last layer
                part2_size = 1 if (part2_max_size == 1 or num_layers2 == 3) else np.random.randint(1, min(part2_max_size, num_layers2 - 2))
                parent2_point11 = 1 if (num_layers2 - part2_size == 1) else np.random.randint(1, num_layers2 - part2_size)
                parent2_point12 = parent2_point11 + part2_size
                    
            # Perform crossover
            new_learning_rate = float(np.random.choice([learning_rate1, learning_rate2]))
            new_batch_size = float(np.random.choice([batch_size1, batch_size2]))
            new_epochs = float(np.random.choice([epochs1, epochs2]))
            new_patience = float(np.random.choice([patience1, patience2]))
            
            new_num_layers = -1
            if parent1_position == 1:
                new_hidden_layer_sizes = np.concatenate((hidden_layer_sizes1, [connection_layer_size[0]], hidden_layer_sizes2[parent2_point11:parent2_point12], [connection_layer_size[1]]))
                new_num_layers = len(new_hidden_layer_sizes)
                new_activations = np.concatenate((activations1, [connection_activation[0]], activations2[parent2_point11:parent2_point12], [connection_activation[1]]))
                new_dropout_rates = np.concatenate((dropout_rates1, [connection_dropout[0]], dropout_rates2[parent2_point11:parent2_point12], [connection_dropout[1]]))
                new_batch_norms = np.concatenate((batch_norms1, [connection_batch_norm[0]], batch_norms2[parent2_point11:parent2_point12], [connection_batch_norm[1]]))
            elif parent1_position == 2:
                new_hidden_layer_sizes = np.concatenate(([connection_layer_size[0]], hidden_layer_sizes2[parent2_point11:parent2_point12], [connection_layer_size[1]], hidden_layer_sizes1))
                new_num_layers = len(new_hidden_layer_sizes)
                new_activations = np.concatenate(([connection_activation[0]], activations2[parent2_point11:parent2_point12], [connection_activation[1]], activations1))
                new_dropout_rates = np.concatenate(([connection_dropout[0]], dropout_rates2[parent2_point11:parent2_point12], [connection_dropout[1]], dropout_rates1))
                new_batch_norms = np.concatenate(([connection_batch_norm[0]], batch_norms2[parent2_point11:parent2_point12], [connection_batch_norm[1]], batch_norms1))
            else:
                new_hidden_layer_sizes = np.concatenate((hidden_layer_sizes1[:parent1_point11], [connection_layer_size[0]], hidden_layer_sizes2[parent2_point11:parent2_point12], [connection_layer_size[1]], hidden_layer_sizes1[parent1_point12:]))
                new_num_layers = len(new_hidden_layer_sizes)
                new_activations = np.concatenate((activations1[:parent1_point11], [connection_activation[0]], activations2[parent2_point11:parent2_point12], [connection_activation[1]], activations1[parent1_point12:]))
                new_dropout_rates = np.concatenate((dropout_rates1[:parent1_point11], [connection_dropout[0]], dropout_rates2[parent2_point11:parent2_point12], [connection_dropout[1]], dropout_rates1[parent1_point12:]))
                new_batch_norms = np.concatenate((batch_norms1[:parent1_point11], [connection_batch_norm[0]], batch_norms2[parent2_point11:parent2_point12], [connection_batch_norm[1]], batch_norms1[parent1_point12:]))

            # Pad each part to MAX_LAYERS length
            new_hidden_layer_sizes = np.append(new_hidden_layer_sizes, [0] * (MAX_LAYERS - new_num_layers))  # 0 indicates padding
            new_activations = np.append(new_activations, [-1] * (MAX_LAYERS - new_num_layers))               # -1 indicates padding
            new_dropout_rates = np.append(new_dropout_rates, [-1.0] * (MAX_LAYERS - new_num_layers))         # -1 indicates padding
            new_batch_norms = np.append(new_batch_norms, [-1] * (MAX_LAYERS - new_num_layers))               # -1 indicates padding
            
            activation_output = parent1[-1] if i == 0 else parent2[-1]
            offspring = np.concatenate((
                [new_learning_rate, new_batch_size, new_epochs, new_patience, new_num_layers],
                new_hidden_layer_sizes.astype(np.float32), 
                new_activations.astype(np.float32), 
                new_dropout_rates.astype(np.float32), 
                new_batch_norms.astype(np.float32), 
                [activation_output]
            ))

            if i == 0:
                offspring1 = offspring
            else:
                offspring2 = offspring

    return offspring1, offspring2



def custom_mutation(offspring, ga_instance):
    # print("... Mutation ...")
    # print("—————————————————————> MUTATION\n")
    for i in range(len(offspring)):
        offspring[i] = structured_mutation(offspring[i])
    return np.array(offspring)


def structured_mutation(individual):
    mutation_probability = 0.02
    
    # Parameters of individual
    learning_rate = float(individual[0])
    batch_size = int(individual[1])
    epochs = int(individual[2])
    patience = int(individual[3])
    num_layers = int(individual[len(BEG_PARAMS)])
    hidden_layer_sizes = individual[len(BEG_PARAMS) + 1:num_layers + len(BEG_PARAMS) + 1].astype(np.int32)
    activations = individual[MAX_LAYERS + len(BEG_PARAMS) + 1:MAX_LAYERS + num_layers + len(BEG_PARAMS) + 1].astype(np.int32)
    dropout_rates = individual[2 * MAX_LAYERS + len(BEG_PARAMS) + 1:2 * MAX_LAYERS + num_layers + len(BEG_PARAMS) + 1]
    batch_norms = individual[3 * MAX_LAYERS + len(BEG_PARAMS) + 1:3 * MAX_LAYERS + num_layers + len(BEG_PARAMS) + 1].astype(np.int32)
    activation_output = individual[-1]
    
    # Mutate number of layers
    if np.random.rand() < mutation_probability:
        num_layers_new = num_layers + 1 if (np.random.rand() < 0.5 or num_layers == 1) and num_layers != MAX_LAYERS else num_layers - 1
        
        if num_layers_new > num_layers:
            # Add new layer
            new_layer_position = np.random.randint(0, num_layers_new)
            
            hidden_layer_sizes = np.insert(hidden_layer_sizes, new_layer_position, np.random.randint(1, MAX_LAYER_SIZE + 1))
            activations = np.insert(activations, new_layer_position, np.random.randint(1, len(ACTIVATIONS) + 1))
            dropout_rates = np.insert(dropout_rates, new_layer_position, np.random.uniform(0.1, 0.5))
            batch_norms = np.insert(batch_norms, new_layer_position, np.random.choice([0, 1]))
            
        elif num_layers_new < num_layers:
            # Remove layer
            layer_to_remove = np.random.randint(0, num_layers)
            
            hidden_layer_sizes = np.delete(hidden_layer_sizes, layer_to_remove)
            activations = np.delete(activations, layer_to_remove)
            dropout_rates = np.delete(dropout_rates, layer_to_remove)
            batch_norms = np.delete(batch_norms, layer_to_remove)
        
        num_layers = num_layers_new
        
    # Mutate learning rate
    if np.random.rand() < mutation_probability:
        learning_rate = np.random.uniform(0.0001, 0.1)
    
    # Mutate batch size
    if np.random.rand() < mutation_probability:
        batch_size = float(np.random.choice([16, 32, 64, 128, 256]))
        
    # Mutate epochs
    if np.random.rand() < mutation_probability:
        epochs = float(np.random.randint(1, 50))
        
    # Mutate patience
    if np.random.rand() < mutation_probability:
        patience = float(np.random.randint(1, 10))
    
    # Mutate hidden layer sizes
    if np.random.rand() < mutation_probability:
        mutation_point = np.random.randint(0, num_layers)
        hidden_layer_sizes[mutation_point] = float(np.random.randint(1, MAX_LAYER_SIZE + 1))
            
    # Mutate activation functions
    if np.random.rand() < mutation_probability:
        mutation_point = np.random.randint(0, num_layers)
        activations[mutation_point] = float(np.random.randint(1, len(ACTIVATIONS) + 1))

    # Mutate dropout rates
    if np.random.rand() < mutation_probability:
        mutation_point = np.random.randint(0, num_layers)
        dropout_rates[mutation_point] = float(np.random.uniform(0.0, 0.5))

    # Mutate batch normalization settings
    if np.random.rand() < mutation_probability:
        mutation_point = np.random.randint(0, num_layers)
        batch_norms[mutation_point] = float(np.random.randint(0, 2))
        
    # Mutate output activation function
    if np.random.rand() < mutation_probability:
        activation_output = float(np.random.randint(1, len(ACTIVATIONS_OUTPUT) + 1))
            
    # Pad each part to MAX_LAYERS length
    num_layers_pad = MAX_LAYERS - num_layers
        
    hidden_layer_sizes = np.append(hidden_layer_sizes, [0] * num_layers_pad)    # 0 indicates padding
    activations = np.append(activations, [-1] * num_layers_pad)                 # -1 indicates padding
    dropout_rates = np.append(dropout_rates, [-1.0] * num_layers_pad)           # -1 indicates padding
    batch_norms = np.append(batch_norms, [-1] * num_layers_pad)                 # -1 indicates padding
        
    individual = np.concatenate(([learning_rate, batch_size, epochs, patience, num_layers], hidden_layer_sizes, activations, dropout_rates, batch_norms, [activation_output]))   

    return individual

    
def save_results_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)

def print_ga_parameters_and_globals(output_dir, ga_index, sol_per_pop, num_generations, num_parents_mating, K_tournaments, keep_parents):
    #global test_size, min_delta, patience_ga, penalty_mult_list, penalty_mult
    params_content = (
        f"Genetic Algorithm Parameters:\n"
        f"Population Size: {sol_per_pop}\n"
        f"Number of Generations: {num_generations}\n"
        f"Number of Parents Mating: {num_parents_mating}\n"
        f"Tournament Size: {K_tournaments}\n"
        f"Number of Parents Kept: {keep_parents}\n\n"
        
        f"Global Variables:\n"
        f"Test Size: {test_size}\n"
        f"Min Delta: {min_delta}\n"
        f"Patience for GA: {patience_ga}\n"
        f"Penalty Multipliers: {penalty_mult_list}\n"
        f"Current Penalty Multiplier: {penalty_mult}\n"
    )
    save_results_to_file(f"{output_dir}/{ga_index+1}_parameters.txt", params_content)

def geneticAlgorithm(ga_index):
    global parent_selection_type, ticks_generation
    # sol_per_pop = 20
    # num_generations = 150
    # num_parents_mating = 3
    # K_tournaments = 2
    # keep_parents = 1

    sol_per_pop = 50
    num_generations = 100
    num_parents_mating = 10
    K_tournaments = 5
    keep_parents = 3

    # sol_per_pop = 100
    # num_generations = 100
    # num_parents_mating = 20
    # K_tournaments = 4
    # keep_parents = 3
    
    parent_selection_type = "tournament"

    ticks_generation = manager.counter(total=num_generations, desc="Generations", unit="gen", color="cyan")
    
    # Print the parameters and global variables to the file
    print_ga_parameters_and_globals(output_dir, ga_index, sol_per_pop, num_generations, num_parents_mating, K_tournaments, keep_parents)
    
    population = generatePopulation(sol_per_pop)
    
    ga_instance = pygad.GA(num_generations=num_generations,
                            num_parents_mating=num_parents_mating,
                            initial_population=population,
                            fitness_func=fitness_func,
                            parent_selection_type=parent_selection_type,
                            K_tournament=K_tournaments,
                            keep_parents=keep_parents,
                            crossover_type=custom_crossover,
                            mutation_type=custom_mutation,
                            on_generation=callback_generation,
                            random_seed=42)

    # Run the genetic algorithm
    ga_instance.run()
    ticks_generation.close()
    # Free memory used by the initial population
    del population
    gc.collect()  # Force garbage collection to free memory
        
    # Plot the fitness history
    # ga_instance.plot_fitness()
    
    return ga_instance

if __name__ == '__main__':
    num_datasets = len(DATASET_LIST)//2
    ticks_dataset = manager.counter(total=num_datasets, desc="Datasets", unit="dataset", color="green")
    
    for dataset_i in range(num_datasets, len(DATASET_LIST)): 
        dataset = DATASET_LIST[dataset_i]
        
        ticks_penalty = manager.counter(total=len(penalty_mult_list), desc="Penalty multipliers", unit="mult", color="blue")
        
        # print(f"————————————————————————————————————————————————————————————")
        # print(f"dataset: {dataset}, index: {dataset_i}")
        # print(f"————————————————————————————————————————————————————————————")
        
        # Load the parameters for the selected dataset
        problem_type, MAX_LAYERS, MAX_LAYER_SIZE, INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, ACTIVATIONS, ACTIVATIONS_OUTPUT = load_and_define_parameters(dataset)
        
        # Set up the output directory
        output_dir = f"./logs/{problem_type}/{dataset}/"
        os.makedirs(output_dir, exist_ok=True)

        # Load and preprocess the dataset
        X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(dataset)

        for i, penalty_mult in enumerate(penalty_mult_list):
            # print(f"——————————————————————————————————————————")
            # print(f"penalty_mult: {penalty_mult}")
            # print(f"——————————————————————————————————————————")
            # if dataset_i == 0 and i == 0:
            #     continue
            start = time.time()
            
            # Run the genetic algorithm for this penalty multiplier
            ga_instance = geneticAlgorithm(i)
            
            # Save the GA instance
            filename = f'{output_dir}genetic{i}'
            ga_instance.save(filename=filename)
            
            # Calculate and format the elapsed time
            end = time.time()
            elapsed_time = time.strftime('%H:%M:%S', time.gmtime(end - start))
            
            # Retrieve the best solution
            solution, solution_fitness, solution_idx = ga_instance.best_solution()
            best_solution = solution
            
            # Convert the best solution to a neural network using the array_to_nn function
            nn2 = array_to_nn(best_solution).to(device)  # Move the model to the GPU

            # Create DataLoader for training and validation
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=int(nn2.batch_size), shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=int(nn2.batch_size), shuffle=False)

            # Train the neural network
            nn2.train_model(train_loader, val_loader)

            # Evaluate the model on training, validation, and test data
            train_loss, train_accuracy = nn2.evaluate(train_loader)
            validation_loss, validation_accuracy = nn2.evaluate(val_loader)
            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=int(nn2.batch_size), shuffle=False)
            test_loss, test_accuracy = nn2.evaluate(test_loader)
            
            # Save the results to a file
            nl = int(best_solution[4])
            act = [ACTIVATIONS[int(a)] for a in best_solution[5+MAX_LAYERS:5+MAX_LAYERS+nl]]
            act_out = ACTIVATIONS_OUTPUT[int(best_solution[-1])]
            results_content = (
                f"Train accuracy: {train_accuracy}\n"
                f"Train loss: {train_loss}\n"
                f"Validation accuracy: {validation_accuracy}\n"
                f"Validation loss: {validation_loss}\n"
                f"Test accuracy: {test_accuracy}\n"
                f"Test loss: {test_loss}\n"
                f"Parameters of the best solution:"
                f"\tLearning rate: {best_solution[0]}\n"
                f"\tBatch size: {best_solution[1]}\n"
                f"\tEpochs: {best_solution[2]}\n"
                f"\tPatience: {best_solution[3]}\n"
                f"\tNumber of layers: {nl}\n"
                f"\tHidden layer sizes: {best_solution[5:5+nl]}\n"
                f"\tActivations: {act}\n"
                f"\tDropout rates: {best_solution[5+2*MAX_LAYERS:5+2*MAX_LAYERS+nl]}\n"
                f"\tBatch normalization: {best_solution[5+3*MAX_LAYERS:5+3*MAX_LAYERS+nl]}\n"
                f"\tOutput activation: {act_out}\n"
                f"Fitness value of the best solution: {solution_fitness}\n"
                f"Index of the best solution: {solution_idx}\n"
                f"Elapsed time: {elapsed_time}\n"
            )
            save_results_to_file(f"{output_dir}/{i+1}_results.txt", results_content)
            
            # Save fitness history per generation
            fitness_history_content = "Generation,Max Fitness,Avg Fitness\n"
            for gen in range(len(fitness_history_best)):
                fitness_history_content += f"{gen + 1},{fitness_history_best[gen]},{fitness_history_avg[gen]}\n"
            save_results_to_file(f"{output_dir}/{i+1}_fitness_history.csv", fitness_history_content)

            # Save fitness history plots
            plt.figure()
            plt.plot(fitness_history_best, label='Best Fitness')
            plt.plot(fitness_history_avg, label='Average Fitness')
            plt.legend()
            plt.title('Fitness per Generation')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.savefig(f"{output_dir}/{i+1}_fitness_plot.jpg")
            plt.close()
            
            # Clear the GA instance to free up memory
            del ga_instance
            torch.cuda.empty_cache()  # Clear GPU memory cache  
            
            ticks_penalty.update()
        ticks_penalty.close()
        ticks_dataset.update()
    ticks_dataset.close()   