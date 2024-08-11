import tensorflow as tf
import numpy as np
import time
import gc
import os
import pygad
import psutil
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_iris, fetch_california_housing, load_diabetes
from sklearn.preprocessing import StandardScaler
from nn2 import *

# Set parameters
test_size = 0.3
min_delta = 0.01  # Minimum change in fitness to qualify as an improvement
patience_ga = 20  # Number of generations to wait before stopping if there is no improvement
penalty_mult_list = [10, 5, 1, 0.5, 0.1, 0.05, 0]  # Penalty multiplier for the complexity of the network

# penalty_mult = -1
fitness_cache = {}
fitness_history_best = []
fitness_history_avg = []
best_fitness = -np.inf
patience_counter = 0
generation_counter = 1
gen_num_printed = True

# Set up logging
logging.basicConfig(filename='memory_usage.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def log_memory_usage(message="Memory usage"):
    process = psutil.Process()
    memory_info = process.memory_info()
    logging.info(f"{message}: RSS={memory_info.rss / (1024 ** 2):.2f} MB, VMS={memory_info.vms / (1024 ** 2):.2f} MB")

def clear_fitness_cache():
    global fitness_cache
    fitness_cache.clear()
    print("Fitness cache cleared.")

def load_and_preprocess_data(dataset):
    if dataset == 'iris':
        iris = load_iris()
        X = iris.data
        y = iris.target
        y = to_categorical(y)

    elif dataset == 'mnist':
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
        X = np.concatenate((X_train_full, X_test)).reshape(-1, 784).astype("float32") / 255
        y = to_categorical(np.concatenate((y_train_full, y_test)), num_classes=10)

    elif dataset == 'california':
        california = fetch_california_housing()
        X = california.data
        y = california.target
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    elif dataset == 'diabetes':
        diabetes = load_diabetes()
        X = diabetes.data
        y = diabetes.target
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_and_get_val_loss(model, X_train, y_train, X_val, y_val, epochs, batch_size, early_stopping):
    history = model.fit(X_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        verbose=0, 
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping])

    validation_loss = history.history['val_loss'][-1]
    return validation_loss

# Callback function for GA
def callback_generation(ga_instance):
    global best_fitness, patience_counter, min_delta, patience_ga, generation_counter, gen_num_printed

    gen_num_printed = False

    best_solution, best_fitness_current, _ = ga_instance.best_solution()
    fitness_history_best.append(best_fitness_current)
    fitness_history_avg.append(np.mean(ga_instance.last_generation_fitness))

    if best_fitness_current - best_fitness > min_delta:
        patience_counter = 0
        best_fitness = best_fitness_current
    else:
        patience_counter += 1

    # Clear the cache after each generation
    clear_fitness_cache()
    tf.keras.backend.clear_session()  
    gc.collect()  

    # Log memory usage
    log_memory_usage(f"After generation {ga_instance.generations_completed + 1}")

    print(f"\n——————————— GENERATION {generation_counter} ———————————\n")
    generation_counter += 1

    if patience_counter >= patience_ga:
        print(f"\nEarly stopping: no improvement in fitness for {patience_ga} generations.\n")
        return "stop"
    

def generatePopulation(sol_per_pop):
    population = []
    for _ in range(sol_per_pop):
        learning_rate = tf.random.uniform([], 0.0001, 0.1, dtype=tf.float32)
        batch_size = tf.cast(tf.random.shuffle([16, 32, 64, 128])[0], tf.float32)
        epochs = tf.cast(tf.random.uniform([], 10, 100, dtype=tf.int32), tf.float32)
        patience = tf.cast(tf.random.uniform([], 1, 10, dtype=tf.int32), tf.float32)

        num_layers = tf.random.uniform([], 1, MAX_LAYERS + 1, dtype=tf.int32)
        hidden_layer_sizes = tf.concat([tf.random.uniform([num_layers], 1, MAX_LAYER_SIZE + 1, dtype=tf.int32),
                                        tf.zeros([MAX_LAYERS - num_layers], dtype=tf.int32)], axis=0)
        activations = tf.concat([tf.random.uniform([num_layers], 1, len(ACTIVATIONS) + 1, dtype=tf.int32),
                                 tf.fill([MAX_LAYERS - num_layers], -1)], axis=0)
        dropout_rates = tf.concat([tf.random.uniform([num_layers], 0.1, 0.5, dtype=tf.float32),
                                   tf.fill([MAX_LAYERS - num_layers], -1.0)], axis=0)
        batch_norms = tf.concat([tf.random.uniform([num_layers], 0, 2, dtype=tf.int32),
                                 tf.fill([MAX_LAYERS - num_layers], -1)], axis=0)
        activation_output = tf.constant(1.0, dtype=tf.float32)

        solution = tf.concat([
            [learning_rate, batch_size, epochs, patience, tf.cast(num_layers, tf.float32)],
            tf.cast(hidden_layer_sizes, tf.float32),
            tf.cast(activations, tf.float32),
            dropout_rates,
            tf.cast(batch_norms, tf.float32),
            [activation_output]
        ], axis=0)

        population.append(solution)

    population = tf.stack(population)
    print("... Initial population generated ...")
    return population

# Custom crossover function
def custom_crossover(parents, offspring_size, ga_instance):
    print("... Crossover ...")
    offspring = []
    for _ in range(offspring_size[0]):
        parent1_idx = np.random.randint(0, len(parents))
        parent2_idx = np.random.randint(0, len(parents))
        parent1 = parents[parent1_idx]
        parent2 = parents[parent2_idx]
        offspring1, offspring2 = structured_crossover(parent1, parent2)
        offspring.append(offspring1)
        offspring.append(offspring2)
    return np.array(offspring[:offspring_size[0]])

def structured_crossover(parent1, parent2):
    # Convert parents to tensors for efficient processing
    parent1 = tf.convert_to_tensor(parent1, dtype=tf.float32)
    parent2 = tf.convert_to_tensor(parent2, dtype=tf.float32)

    # Extract parameters from parents
    learning_rate1 = parent1[0]
    batch_size1 = parent1[1]
    epochs1 = parent1[2]
    patience1 = parent1[3]
    num_layers1 = tf.cast(parent1[len(BEG_PARAMS)], tf.int32)
    hidden_layer_sizes1 = tf.cast(parent1[len(BEG_PARAMS) + 1:num_layers1 + len(BEG_PARAMS) + 1], tf.int32)
    activations1 = tf.cast(parent1[MAX_LAYERS + len(BEG_PARAMS) + 1:MAX_LAYERS + num_layers1 + len(BEG_PARAMS) + 1], tf.int32)
    dropout_rates1 = parent1[2 * MAX_LAYERS + len(BEG_PARAMS) + 1:2 * MAX_LAYERS + num_layers1 + len(BEG_PARAMS) + 1]
    batch_norms1 = tf.cast(parent1[3 * MAX_LAYERS + len(BEG_PARAMS) + 1:3 * MAX_LAYERS + num_layers1 + len(BEG_PARAMS) + 1], tf.int32)

    learning_rate2 = parent2[0]
    batch_size2 = parent2[1]
    epochs2 = parent2[2]
    patience2 = parent2[3]
    num_layers2 = tf.cast(parent2[len(BEG_PARAMS)], tf.int32)
    hidden_layer_sizes2 = tf.cast(parent2[len(BEG_PARAMS) + 1:num_layers2 + len(BEG_PARAMS) + 1], tf.int32)
    activations2 = tf.cast(parent2[MAX_LAYERS + len(BEG_PARAMS) + 1:MAX_LAYERS + num_layers2 + len(BEG_PARAMS) + 1], tf.int32)
    dropout_rates2 = parent2[2 * MAX_LAYERS + len(BEG_PARAMS) + 1:2 * MAX_LAYERS + num_layers2 + len(BEG_PARAMS) + 1]
    batch_norms2 = tf.cast(parent2[3 * MAX_LAYERS + len(BEG_PARAMS) + 1:3 * MAX_LAYERS + num_layers2 + len(BEG_PARAMS) + 1], tf.int32)

    # Select crossover strategy
    cross_option = tf.random.uniform((), minval=1, maxval=3, dtype=tf.int32)

    if cross_option == 1:
        # Option 1: Mix and match layers from both parents with a connection layer in between

        # Select crossover points
        if num_layers1 == 1:
            parent1_point1 = 1
            parent1_point2 = 0
        else:
            parent1_point1 = tf.random.uniform((), minval=1, maxval=min(num_layers1, MAX_LAYERS - 2), dtype=tf.int32)

        if num_layers2 == 1:
            parent2_point1 = 0
            parent2_point2 = 1
        else:
            parent2_point1 = tf.random.uniform((), minval=max(0, num_layers2 - (MAX_LAYERS - parent1_point1 - 1)), maxval=num_layers2, dtype=tf.int32)
            parent2_point2 = tf.random.uniform((), minval=1, maxval=min(num_layers2, MAX_LAYERS - 2), dtype=tf.int32)

        if parent1_point2 == -1:
            parent1_point2 = tf.random.uniform((), minval=max(0, num_layers1 - (MAX_LAYERS - parent2_point2 - 1)), maxval=num_layers1, dtype=tf.int32)

        connection_layer_size = tf.random.uniform((2,), minval=1, maxval=MAX_LAYER_SIZE + 1, dtype=tf.int32)
        connection_activation = tf.random.uniform((2,), minval=1, maxval=len(ACTIVATIONS) + 1, dtype=tf.int32)
        connection_dropout = tf.random.uniform((2,), minval=0.1, maxval=0.5)
        connection_batch_norm = tf.random.uniform((2,), minval=0, maxval=2, dtype=tf.int32)

        new_hidden_layer_sizes1 = tf.concat([hidden_layer_sizes1[:parent1_point1], [connection_layer_size[0]], hidden_layer_sizes2[parent2_point1:]], axis=0)
        new_hidden_layer_sizes2 = tf.concat([hidden_layer_sizes2[:parent2_point2], [connection_layer_size[1]], hidden_layer_sizes1[parent1_point2:]], axis=0)

        new_num_layers1 = tf.size(new_hidden_layer_sizes1)
        new_num_layers2 = tf.size(new_hidden_layer_sizes2)

        new_activations1 = tf.concat([activations1[:parent1_point1], [connection_activation[0]], activations2[parent2_point1:]], axis=0)
        new_activations2 = tf.concat([activations2[:parent2_point2], [connection_activation[1]], activations1[parent1_point2:]], axis=0)

        new_dropout_rates1 = tf.concat([dropout_rates1[:parent1_point1], [connection_dropout[0]], dropout_rates2[parent2_point1:]], axis=0)
        new_dropout_rates2 = tf.concat([dropout_rates2[:parent2_point2], [connection_dropout[1]], dropout_rates1[parent1_point2:]], axis=0)

        new_batch_norms1 = tf.concat([batch_norms1[:parent1_point1], [connection_batch_norm[0]], batch_norms2[parent2_point1:]], axis=0)
        new_batch_norms2 = tf.concat([batch_norms2[:parent2_point2], [connection_batch_norm[1]], batch_norms1[parent1_point2:]], axis=0)

    else:
        # Option 2: Begin with parent 1, mix in a connection layer from parent 2, and end with parent 1
        offspring1 = []
        offspring2 = []
        for i in range(2):
            if i == 1:
                learning_rate1 = parent1[0]
                batch_size1 = parent1[1]
                epochs1 = parent1[2]
                patience1 = parent1[3]
                num_layers1 = tf.cast(parent1[len(BEG_PARAMS)], tf.int32)
                hidden_layer_sizes1 = tf.cast(parent1[len(BEG_PARAMS) + 1:num_layers1 + len(BEG_PARAMS) + 1], tf.int32)
                activations1 = tf.cast(parent1[MAX_LAYERS + len(BEG_PARAMS) + 1:MAX_LAYERS + num_layers1 + len(BEG_PARAMS) + 1], tf.int32)
                dropout_rates1 = parent1[2 * MAX_LAYERS + len(BEG_PARAMS) + 1:2 * MAX_LAYERS + num_layers1 + len(BEG_PARAMS) + 1]
                batch_norms1 = tf.cast(parent1[3 * MAX_LAYERS + len(BEG_PARAMS) + 1:3 * MAX_LAYERS + num_layers1 + len(BEG_PARAMS) + 1], tf.int32)

                learning_rate2 = parent2[0]
                batch_size2 = parent2[1]
                epochs2 = parent2[2]
                patience2 = parent2[3]
                num_layers2 = tf.cast(parent2[len(BEG_PARAMS)], tf.int32)
                hidden_layer_sizes2 = tf.cast(parent2[len(BEG_PARAMS) + 1:num_layers2 + len(BEG_PARAMS) + 1], tf.int32)
                activations2 = tf.cast(parent2[MAX_LAYERS + len(BEG_PARAMS) + 1:MAX_LAYERS + num_layers2 + len(BEG_PARAMS) + 1], tf.int32)
                dropout_rates2 = parent2[2 * MAX_LAYERS + len(BEG_PARAMS) + 1:2 * MAX_LAYERS + num_layers2 + len(BEG_PARAMS) + 1]
                batch_norms2 = tf.cast(parent2[3 * MAX_LAYERS + len(BEG_PARAMS) + 1:3 * MAX_LAYERS + num_layers2 + len(BEG_PARAMS) + 1], tf.int32)
                
            if num_layers1 == 1:
                # If parent 1 has only one layer, randomly choose whether to place it at the beginning or end
                parent1_position = 1 if tf.random.uniform((), 0, 1) < 0.5 else 2  # 1 for beginning, 2 for end
                part1_size = 1
            else:
                parent1_position = 0  # 0 for default
                if num_layers1 == 2:
                    parent1_point11 = parent1_point12 = 1
                    part1_size = 2
                else:
                    parent1_point11 = tf.random.uniform((), minval=1, maxval=min(num_layers1 - 1, MAX_LAYERS - 4), dtype=tf.int32)
                    parent1_point12 = tf.random.uniform((), minval=max(parent1_point11, num_layers1 - (MAX_LAYERS - parent1_point11 - 3)), maxval=num_layers1, dtype=tf.int32)
                    part1_size = parent1_point11 + (num_layers1 - parent1_point12)

            part2_max_size = MAX_LAYERS - part1_size - 2

            if num_layers2 == 1:
                parent2_point11 = 0
                parent2_point12 = 1
            elif num_layers2 == 2:
                part2_size = min(part2_max_size, tf.random.uniform((), minval=1, maxval=3, dtype=tf.int32))
                if part2_size == 1:
                    parent2_point11 = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32)
                    parent2_point12 = parent2_point11 + 1
                else:
                    parent2_point11 = 0
                    parent2_point12 = 2
            else:
                part2_size = tf.random.uniform((), minval=1, maxval=min(part2_max_size, num_layers2 - 2), dtype=tf.int32)
                parent2_point11 = tf.random.uniform((), minval=1, maxval=num_layers2 - part2_size, dtype=tf.int32)
                parent2_point12 = parent2_point11 + part2_size

            connection_layer_size = tf.random.uniform((2,), minval=1, maxval=MAX_LAYER_SIZE + 1, dtype=tf.int32)
            connection_activation = tf.random.uniform((2,), minval=1, maxval=len(ACTIVATIONS) + 1, dtype=tf.int32)
            connection_dropout = tf.random.uniform((2,), minval=0.1, maxval=0.5)
            connection_batch_norm = tf.random.uniform((2,), minval=0, maxval=2, dtype=tf.int32)

            if parent1_position == 1:
                new_hidden_layer_sizes = tf.concat([hidden_layer_sizes1, [connection_layer_size[0]], hidden_layer_sizes2[parent2_point11:parent2_point12], [connection_layer_size[1]]], axis=0)
                new_activations = tf.concat([activations1, [connection_activation[0]], activations2[parent2_point11:parent2_point12], [connection_activation[1]]], axis=0)
                new_dropout_rates = tf.concat([dropout_rates1, [connection_dropout[0]], dropout_rates2[parent2_point11:parent2_point12], [connection_dropout[1]]], axis=0)
                new_batch_norms = tf.concat([batch_norms1, [connection_batch_norm[0]], batch_norms2[parent2_point11:parent2_point12], [connection_batch_norm[1]]], axis=0)
            elif parent1_position == 2:
                new_hidden_layer_sizes = tf.concat([[connection_layer_size[0]], hidden_layer_sizes2[parent2_point11:parent2_point12], [connection_layer_size[1]], hidden_layer_sizes1], axis=0)
                new_activations = tf.concat([[connection_activation[0]], activations2[parent2_point11:parent2_point12], [connection_activation[1]], activations1], axis=0)
                new_dropout_rates = tf.concat([[connection_dropout[0]], dropout_rates2[parent2_point11:parent2_point12], [connection_dropout[1]], dropout_rates1], axis=0)
                new_batch_norms = tf.concat([[connection_batch_norm[0]], batch_norms2[parent2_point11:parent2_point12], [connection_batch_norm[1]], batch_norms1], axis=0)
            else:
                new_hidden_layer_sizes = tf.concat([hidden_layer_sizes1[:parent1_point11], [connection_layer_size[0]], hidden_layer_sizes2[parent2_point11:parent2_point12], [connection_layer_size[1]], hidden_layer_sizes1[parent1_point12:]], axis=0)
                new_activations = tf.concat([activations1[:parent1_point11], [connection_activation[0]], activations2[parent2_point11:parent2_point12], [connection_activation[1]], activations1[parent1_point12:]], axis=0)
                new_dropout_rates = tf.concat([dropout_rates1[:parent1_point11], [connection_dropout[0]], dropout_rates2[parent2_point11:parent2_point12], [connection_dropout[1]], dropout_rates1[parent1_point12:]], axis=0)
                new_batch_norms = tf.concat([batch_norms1[:parent1_point11], [connection_batch_norm[0]], batch_norms2[parent2_point11:parent2_point12], [connection_batch_norm[1]], batch_norms1[parent1_point12:]], axis=0)

            new_num_layers = tf.size(new_hidden_layer_sizes)

            # Finalize offspring
            new_hidden_layer_sizes = tf.pad(new_hidden_layer_sizes, [[0, MAX_LAYERS - new_num_layers]], constant_values=0)
            new_activations = tf.pad(new_activations, [[0, MAX_LAYERS - new_num_layers]], constant_values=-1)
            new_dropout_rates = tf.pad(new_dropout_rates, [[0, MAX_LAYERS - new_num_layers]], constant_values=-1.0)
            new_batch_norms = tf.pad(new_batch_norms, [[0, MAX_LAYERS - new_num_layers]], constant_values=-1)

            offspring = tf.concat([[learning_rate1, batch_size1, epochs1, patience1, new_num_layers], 
                                    tf.cast(new_hidden_layer_sizes, tf.float32), 
                                    tf.cast(new_activations, tf.float32), 
                                    new_dropout_rates, 
                                    tf.cast(new_batch_norms, tf.float32), 
                                    [parent1[-1]]], axis=0).numpy()
            if i == 0:
                offspring1 = offspring
            else:
                offspring2 = offspring
        return offspring1, offspring2

    # Finalize offspring
    new_hidden_layer_sizes1 = tf.pad(new_hidden_layer_sizes1, [[0, MAX_LAYERS - new_num_layers1]])
    new_hidden_layer_sizes2 = tf.pad(new_hidden_layer_sizes2, [[0, MAX_LAYERS - new_num_layers2]])
    new_activations1 = tf.pad(new_activations1, [[0, MAX_LAYERS - new_num_layers1]], constant_values=-1)
    new_activations2 = tf.pad(new_activations2, [[0, MAX_LAYERS - new_num_layers2]], constant_values=-1)
    new_dropout_rates1 = tf.pad(new_dropout_rates1, [[0, MAX_LAYERS - new_num_layers1]], constant_values=-1.0)
    new_dropout_rates2 = tf.pad(new_dropout_rates2, [[0, MAX_LAYERS - new_num_layers2]], constant_values=-1.0)
    new_batch_norms1 = tf.pad(new_batch_norms1, [[0, MAX_LAYERS - new_num_layers1]], constant_values=-1)
    new_batch_norms2 = tf.pad(new_batch_norms2, [[0, MAX_LAYERS - new_num_layers2]], constant_values=-1)

    offspring1 = tf.concat([[learning_rate1, batch_size1, epochs1, patience1, new_num_layers1], 
                            tf.cast(new_hidden_layer_sizes1, tf.float32), 
                            tf.cast(new_activations1, tf.float32), 
                            new_dropout_rates1, 
                            tf.cast(new_batch_norms1, tf.float32), 
                            [parent1[-1]]], axis=0).numpy()

    offspring2 = tf.concat([[learning_rate2, batch_size2, epochs2, patience2, new_num_layers2], 
                            tf.cast(new_hidden_layer_sizes2, tf.float32), 
                            tf.cast(new_activations2, tf.float32), 
                            new_dropout_rates2, 
                            tf.cast(new_batch_norms2, tf.float32), 
                            [parent2[-1]]], axis=0).numpy()

    return offspring1, offspring2


# Custom mutation function
def custom_mutation(offspring, ga_instance):
    print("... Mutation ...")
    for i in range(len(offspring)):
        offspring[i] = structured_mutation(offspring[i])
    return np.array(offspring)

def structured_mutation(individual):
    mutation_probability = 0.02
    mutated = False
    
    # Parameters of individual (converted to tensors)
    learning_rate = tf.convert_to_tensor(float(individual[0]), dtype=tf.float32)
    batch_size = tf.convert_to_tensor(int(individual[1]), dtype=tf.int32)
    epochs = tf.convert_to_tensor(int(individual[2]), dtype=tf.int32)
    patience = tf.convert_to_tensor(int(individual[3]), dtype=tf.int32)
    num_layers = tf.convert_to_tensor(int(individual[len(BEG_PARAMS)]), dtype=tf.int32)
    hidden_layer_sizes = tf.convert_to_tensor(individual[len(BEG_PARAMS) + 1:num_layers + len(BEG_PARAMS) + 1], dtype=tf.int32)
    activations = tf.convert_to_tensor(individual[MAX_LAYERS + len(BEG_PARAMS) + 1:MAX_LAYERS + num_layers + len(BEG_PARAMS) + 1], dtype=tf.int32)
    dropout_rates = tf.convert_to_tensor(individual[2 * MAX_LAYERS + len(BEG_PARAMS) + 1:2 * MAX_LAYERS + num_layers + len(BEG_PARAMS) + 1], dtype=tf.float32)
    batch_norms = tf.convert_to_tensor(individual[3 * MAX_LAYERS + len(BEG_PARAMS) + 1:3 * MAX_LAYERS + num_layers + len(BEG_PARAMS) + 1], dtype=tf.int32)
    
    # Mutate number of layers
    if tf.random.uniform(()) < mutation_probability:
        mutated = True
        num_layers_new = num_layers + 1 if (tf.random.uniform(()) < 0.5 or num_layers == 1) and num_layers != MAX_LAYERS else num_layers - 1
        
        if num_layers_new > num_layers:
            # Add new layer
            new_layer_position = tf.random.uniform((), minval=0, maxval=num_layers_new, dtype=tf.int32)
            hidden_layer_sizes = tf.concat([hidden_layer_sizes[:new_layer_position], 
                                            [tf.random.uniform((), minval=1, maxval=MAX_LAYER_SIZE + 1, dtype=tf.int32)], 
                                            hidden_layer_sizes[new_layer_position:]], axis=0)
            activations = tf.concat([activations[:new_layer_position], 
                                     [tf.random.uniform((), minval=1, maxval=len(ACTIVATIONS) + 1, dtype=tf.int32)], 
                                     activations[new_layer_position:]], axis=0)
            dropout_rates = tf.concat([dropout_rates[:new_layer_position], 
                                       [tf.random.uniform((), minval=0.1, maxval=0.5)], 
                                       dropout_rates[new_layer_position:]], axis=0)
            batch_norms = tf.concat([batch_norms[:new_layer_position], 
                                     [tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32)], 
                                     batch_norms[new_layer_position:]], axis=0)
            
        elif num_layers_new < num_layers:
            # Remove layer
            layer_to_remove = tf.random.uniform((), minval=0, maxval=num_layers, dtype=tf.int32)
            hidden_layer_sizes = tf.concat([hidden_layer_sizes[:layer_to_remove], hidden_layer_sizes[layer_to_remove + 1:]], axis=0)
            activations = tf.concat([activations[:layer_to_remove], activations[layer_to_remove + 1:]], axis=0)
            dropout_rates = tf.concat([dropout_rates[:layer_to_remove], dropout_rates[layer_to_remove + 1:]], axis=0)
            batch_norms = tf.concat([batch_norms[:layer_to_remove], batch_norms[layer_to_remove + 1:]], axis=0)
        
        num_layers = num_layers_new

    # Mutate learning rate
    if tf.random.uniform(()) < mutation_probability:
        mutated = True
        learning_rate = tf.random.uniform((), minval=0.0001, maxval=0.1, dtype=tf.float32)
    
    # Mutate batch size
    if tf.random.uniform(()) < mutation_probability:
        mutated = True
        batch_size = tf.convert_to_tensor(float(tf.random.choice([16, 32, 64, 128])), dtype=tf.float32)
        
    # Mutate epochs
    if tf.random.uniform(()) < mutation_probability:
        mutated = True
        epochs = tf.random.uniform((), minval=10, maxval=100, dtype=tf.int32)
        
    # Mutate patience
    if tf.random.uniform(()) < mutation_probability:
        mutated = True
        patience = tf.random.uniform((), minval=1, maxval=10, dtype=tf.int32)
    
    # Mutate hidden layer sizes
    if tf.random.uniform(()) < mutation_probability:
        mutated = True
        mutation_point = tf.random.uniform((), minval=0, maxval=num_layers, dtype=tf.int32)
        hidden_layer_sizes = tf.tensor_scatter_nd_update(hidden_layer_sizes, [[mutation_point]], 
                                                         [tf.random.uniform((), minval=1, maxval=MAX_LAYER_SIZE + 1, dtype=tf.int32)])
            
    # Mutate activation functions
    if tf.random.uniform(()) < mutation_probability:
        mutated = True
        mutation_point = tf.random.uniform((), minval=0, maxval=num_layers, dtype=tf.int32)
        activations = tf.tensor_scatter_nd_update(activations, [[mutation_point]], 
                                                  [tf.random.uniform((), minval=1, maxval=len(ACTIVATIONS) + 1, dtype=tf.int32)])

    # Mutate dropout rates
    if tf.random.uniform(()) < mutation_probability:
        mutated = True
        mutation_point = tf.random.uniform((), minval=0, maxval=num_layers, dtype=tf.int32)
        dropout_rates = tf.tensor_scatter_nd_update(dropout_rates, [[mutation_point]], 
                                                    [tf.random.uniform((), minval=0.1, maxval=0.5)])

    # Mutate batch normalization settings
    if tf.random.uniform(()) < mutation_probability:
        mutated = True
        mutation_point = tf.random.uniform((), minval=0, maxval=num_layers, dtype=tf.int32)
        batch_norms = tf.tensor_scatter_nd_update(batch_norms, [[mutation_point]], 
                                                  [tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32)])
            
    # Pad each part to MAX_LAYERS length
    num_layers_pad = MAX_LAYERS - num_layers
    
    hidden_layer_sizes = tf.concat([hidden_layer_sizes, tf.zeros(num_layers_pad, dtype=tf.int32)], axis=0)
    activations = tf.concat([activations, -tf.ones(num_layers_pad, dtype=tf.int32)], axis=0)
    dropout_rates = tf.concat([dropout_rates, -tf.ones(num_layers_pad, dtype=tf.float32)], axis=0)
    batch_norms = tf.concat([batch_norms, -tf.ones(num_layers_pad, dtype=tf.int32)], axis=0)
    
    individual = tf.concat([[learning_rate, batch_size, epochs, patience, num_layers], 
                            tf.cast(hidden_layer_sizes, tf.float32), 
                            tf.cast(activations, tf.float32), 
                            dropout_rates, 
                            tf.cast(batch_norms, tf.float32), 
                            [individual[-1]]], axis=0)
    
    # If any mutation occurs, remove the individual from the cache
    if mutated:
        solution_id = tuple(individual.numpy())
        if solution_id in fitness_cache:
            del fitness_cache[solution_id]

    return individual.numpy()


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

def geneticAlgorithm(output_dir, ga_index):
    sol_per_pop = 5
    num_generations = 100
    num_parents_mating = 3
    K_tournaments = 2
    keep_parents = 1
    
    # Print the parameters and global variables to the file
    print_ga_parameters_and_globals(output_dir, ga_index, sol_per_pop, num_generations, num_parents_mating, K_tournaments, keep_parents)
    
    population = generatePopulation(sol_per_pop)
    
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           initial_population=population,
                           fitness_func=fitness_func,
                           parent_selection_type="tournament",
                           K_tournament=K_tournaments,
                           keep_parents=keep_parents,
                           crossover_type=custom_crossover,
                           mutation_type=custom_mutation,
                           on_generation=callback_generation,
                           random_seed=42)

    ga_instance.run()
    
    return ga_instance

if __name__ == '__main__':
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # dataset = 'mnist'
    output_dir = f"./logs/{problem_type}/{dataset}/"
    os.makedirs(output_dir, exist_ok=True)
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(dataset)
 

    for i, penalty_mult in enumerate(penalty_mult_list):
        start = time.time()
        ga_instance = geneticAlgorithm(output_dir, i)
        
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        best_solution = solution
        
        nn2 = array_to_nn(best_solution)

        nn2.model.fit(X_train, y_train, 
                      epochs=int(nn2.epochs), 
                      batch_size=int(nn2.batch_size), 
                      validation_data=(X_val, y_val),
                      callbacks=[nn2.early_stopping])

        test_loss, test_accuracy = nn2.model.evaluate(X_test, y_test)
        validation_loss, validation_accuracy = nn2.model.evaluate(X_val, y_val)
        train_loss, train_accuracy = nn2.model.evaluate(X_train, y_train)

        end = time.time()
        elapsed_time = time.strftime('%H:%M:%S', time.gmtime(end - start))
        
        # Save the results to a file
        results_content = (
            f"Train accuracy: {train_accuracy}\n"
            f"Train loss: {train_loss}\n"
            f"Validation accuracy: {validation_accuracy}\n"
            f"Validation loss: {validation_loss}\n"
            f"Test accuracy: {test_accuracy}\n"
            f"Test loss: {test_loss}\n"
            f"Parameters of the best solution: {solution}\n"
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

        # Clear memory after each generation
        del ga_instance
        del nn2
        tf.keras.backend.clear_session()
        gc.collect()