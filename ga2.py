from re import T
import sys
import time
from matplotlib.pylab import rand
import matplotlib.pyplot as plt
import numpy as np
import pygad
from sklearn.datasets import load_iris, fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from nn2 import *


test_size = 0.3
min_delta = 0.01  # Minimum change in fitness to qualify as an improvement
patience_ga = 10  # Number of generations to wait before stopping if there is no improvement
penalty_mult_list = [0, 0.01, 0.05, 0.1, 0.5, 1, 2, 5]  # Penalty multiplier for the complexity of the network
penalty_mult = -1

fitness_history_best = []
fitness_history_avg = []
best_fitness = -np.inf
patience_counter = 0
generation_counter = 1

gen_num_printed = True

def callback_generation(ga_instance):
    global best_fitness, patience_counter, min_delta, patience_ga, generation_counter, gen_num_printed
    
    gen_num_printed = False
        
    # Save the fitness score for the best, the worst, and the average solution in each generation
    best_solution, best_fitness_current, _ = ga_instance.best_solution()
    fitness_history_best.append(best_fitness_current)
    fitness_history_avg.append(np.mean(ga_instance.last_generation_fitness))

    # Early stopping logic
    if best_fitness_current - best_fitness > min_delta:
        patience_counter = 0
        best_fitness = best_fitness_current
    else:
        patience_counter += 1

    if patience_counter >= patience_ga:
        print(f"\nEarly stopping: no improvement in fitness for {patience_ga} generations.\n")
        return "stop"
        
def generatePopulation(sol_per_pop):
    population = np.array([])
    for _ in range(sol_per_pop):
        # Generate random solution parts
        learning_rate = np.random.uniform(0.0001, 0.1)
        batch_size = float(np.random.randint(16, 128))
        epochs = float(np.random.randint(10, 100))
        patience = float(np.random.randint(1, 10))
        
        num_layers = np.random.randint(1, MAX_LAYERS + 1)
        hidden_layer_sizes = [np.random.randint(1, MAX_LAYER_SIZE + 1) for _ in range(num_layers)]
        activations = [np.random.randint(1, len(ACTIVATIONS) + 1) for _ in range(num_layers)]
        dropout_rates = [np.random.uniform(0.1, 0.5) for _ in range(num_layers)]
        batch_norms = [np.random.choice([0, 1]) for _ in range(num_layers)]
        activation_output = float(np.random.randint(1, len(ACTIVATIONS_OUTPUT) + 1))

        # Pad each part to MAX_LAYERS length
        hidden_layer_sizes = np.append(hidden_layer_sizes, [0] * (MAX_LAYERS - num_layers))  # 0 indicates padding
        activations = np.append(activations, [-1] * (MAX_LAYERS - num_layers))               # -1 indicates padding
        dropout_rates = np.append(dropout_rates, [-1.0] * (MAX_LAYERS - num_layers))         # -1 indicates padding
        batch_norms = np.append(batch_norms, [-1] * (MAX_LAYERS - num_layers))               # -1 indicates padding
        
        # Combine all parts into a single solution array
        solution = np.concatenate((
                [learning_rate, batch_size, epochs, patience, num_layers],
                hidden_layer_sizes.astype(np.float32), 
                activations.astype(np.float32), 
                dropout_rates.astype(np.float32), 
                batch_norms.astype(np.float32), 
                [activation_output] 
            ))
                    
        # Check if population is empty
        if population.size == 0:
            # Initialize population with solution as the first nested array
            population = np.array([solution])
        else:
            # Concatenate solution as a new row to the existing population
            population = np.vstack((population, solution))
        
    # print("————————————————————> POPULATION: ", population)
    print("\n——————————— GENERATION 0 ———————————\n")
    return population

def fitness_func(ga_instance, solution, solution_idx):
    global gen_num_printed
    
    # Create a neural network from the solution array
    solution_nn = array_to_nn(solution)
    
    # Train the neural network
    history = solution_nn.model.fit(X_train, y_train, 
                                    epochs=int(solution_nn.epochs), 
                                    batch_size=int(solution_nn.batch_size), 
                                    verbose=0, 
                                    validation_data=(X_test, y_test),
                                    callbacks=[solution_nn.early_stopping])
    
    # Get the validation accuracy and loss
    validation_loss = history.history['val_loss'][-1]
    if problem_type == "classification":
        validation_accuracy = history.history['val_accuracy'][-1]
    
    # Calculate the number of layers and total number of neurons
    num_layers = solution_nn.num_layers
    total_neurons = np.sum(solution_nn.hidden_layer_sizes)
    
    # Calculate the relative complexity of the network
    relative_layers = num_layers / MAX_LAYERS
    relative_neurons = total_neurons / (MAX_LAYERS * MAX_LAYER_SIZE)  # Maximum possible neurons
    
    layer_mult = 0.1
    neuron_mult = 0.01
    
    # Calculate the penalty based on relative complexity
    penalty = relative_layers * layer_mult + relative_neurons * neuron_mult
    
    # Calculate the fitness score
    fitness_score = 1 / (validation_loss + penalty_mult * penalty)
    if not gen_num_printed:
        print(f"——————————— GENERATION {ga_instance.generations_completed + 1} ———————————\n")
        print("population:\n", ga_instance.population)
        gen_num_printed = True
    
    print("individual:\n", solution)
    
    if problem_type == "classification":
        print(f"—> validation accuracy: {validation_accuracy}")
    print(f"—> validation loss: {validation_loss}")
    print(f"—> fitness score = 1 / (validation_loss + penalty_mult * penalty) = 1 / ({validation_loss} + {penalty_mult} * {penalty}) = {fitness_score}\n")
    
    return fitness_score

def custom_crossover(parents, offspring_size, ga_instance):
    print("... Crossover ...")
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
    cross_option = 2
    # print(f"——> crossover option: {cross_option}\n")
    # print(f"—> parent 1:\n{parent1}")
    # print(f"—> parent 2:\n{parent2}\n")
    
    # OPTION 1: beginning of parent 1 + "connection layer" + end of parent 2
    if cross_option == 1:
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
            [parent1[-1]]
        ))
        
        offspring2 = np.concatenate((
            [new_learning_rate2, new_batch_size2, new_epochs2, new_patience2, new_num_layers2],
            new_hidden_layer_sizes2.astype(np.float32), 
            new_activations2.astype(np.float32), 
            new_dropout_rates2.astype(np.float32), 
            new_batch_norms2.astype(np.float32), 
            [parent2[-1]]
        ))
        
    # OPTION 2: beginning of parent 1 + "connection layer" + middle of parent 2 + "connection layer" + end of parent 1
    elif cross_option == 2:
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
            
            offspring = np.concatenate((
                [new_learning_rate, new_batch_size, new_epochs, new_patience, new_num_layers],
                new_hidden_layer_sizes.astype(np.float32), 
                new_activations.astype(np.float32), 
                new_dropout_rates.astype(np.float32), 
                new_batch_norms.astype(np.float32), 
                [parent1[-1] if i == 0 else parent2[-1]] 
            ))
            
            # print(f"—> offspring {i}:\n{offspring}\n")
            # print(f"new_learning_rate: {new_learning_rate}")
            # print(f"new_batch_size: {new_batch_size}")
            # print(f"new_epochs: {new_epochs}")
            # print(f"new_patience: {new_patience}")
            # print(f"new_num_layers: {new_num_layers}")
            # print(f"new_hidden_layer_sizes: {new_hidden_layer_sizes}")
            # print(f"new_activations: {new_activations}")
            # print(f"new_dropout_rates: {new_dropout_rates}")
            # print(f"new_batch_norms: {new_batch_norms}")
        
            if i == 0:
                offspring1 = offspring
            else:
                offspring2 = offspring

    # print(f"\n———> OFFSPRING:")
    # print(f"—> offspring 1:\n{offspring1}\n")
    # print(f"—> offspring 2:\n{offspring2}")
    return offspring1, offspring2

def custom_mutation(offspring, ga_instance):
    print("... Mutation ...")
    # print("—————————————————————> MUTATION\n")
    for i in range(len(offspring)):
        offspring[i] = structured_mutation(offspring[i])
    return np.array(offspring)

def structured_mutation(individual):
    mutation_probability = 0.02
    # print(f"——> INDIVIDUAL:\n{individual}\n")
    
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
    
    # Mutate number of layers
    if np.random.rand() < mutation_probability:
        num_layers_new = num_layers + 1 if (np.random.rand() < 0.5 or num_layers == 1) and num_layers != MAX_LAYERS else num_layers - 1
        # print(f"——> mutated number of layers: {num_layers} -> {num_layers_new}")
        
        if num_layers_new > num_layers:
            # Add new layer
            new_layer_position = np.random.randint(0, num_layers_new)
            # print(f"   —> added layer index: {new_layer_position}")
            
            hidden_layer_sizes = np.insert(hidden_layer_sizes, new_layer_position, np.random.randint(1, MAX_LAYER_SIZE + 1))
            activations = np.insert(activations, new_layer_position, np.random.randint(1, len(ACTIVATIONS) + 1))
            dropout_rates = np.insert(dropout_rates, new_layer_position, np.random.uniform(0.1, 0.5))
            batch_norms = np.insert(batch_norms, new_layer_position, np.random.choice([0, 1]))
            
        elif num_layers_new < num_layers:
            # Remove layer
            layer_to_remove = np.random.randint(0, num_layers)
            # print(f"—> deleted layer index: {layer_to_remove}")
            
            hidden_layer_sizes = np.delete(hidden_layer_sizes, layer_to_remove)
            activations = np.delete(activations, layer_to_remove)
            dropout_rates = np.delete(dropout_rates, layer_to_remove)
            batch_norms = np.delete(batch_norms, layer_to_remove)
        
        num_layers = num_layers_new
        
    # Mutate learning rate
    if np.random.rand() < mutation_probability:
        learning_rate_old = learning_rate
        while learning_rate_old == learning_rate:
            learning_rate = np.random.uniform(0.0001, 0.1)
        # print(f"——> mutated learning rate: {learning_rate}")
    
    # Mutate batch size
    if np.random.rand() < mutation_probability:
        batch_size_old = batch_size
        while batch_size_old == batch_size:
            batch_size = float(np.random.randint(16, 128))
        # print(f"——> mutated batch size: {batch_size}")
        
    # Mutate epochs
    if np.random.rand() < mutation_probability:
        epochs_old = epochs
        while epochs_old == epochs:
            epochs = float(np.random.randint(10, 100))
        # print(f"——> mutated epochs: {epochs}")
        
    # Mutate patience
    if np.random.rand() < mutation_probability:
        patience_old = patience
        while patience_old == patience:
            patience = float(np.random.randint(1, 10))
        # print(f"——> mutated patience: {patience}")
    
    # Mutate hidden layer sizes
    if np.random.rand() < mutation_probability:
        mutation_point = np.random.randint(0, num_layers)
        hidden_layer_size_old = hidden_layer_sizes[mutation_point]
        while hidden_layer_size_old == hidden_layer_sizes[mutation_point]:
            hidden_layer_sizes[mutation_point] = float(np.random.randint(1, MAX_LAYER_SIZE + 1))
        # print(f"——> mutated hidden layer size at index {mutation_point}: {hidden_layer_sizes[mutation_point]}")
            
    # Mutate activation functions
    if np.random.rand() < mutation_probability:
        mutation_point = np.random.randint(0, num_layers)
        activation_old = activations[mutation_point]
        while activation_old == activations[mutation_point]:
            activations[mutation_point] = float(np.random.randint(1, len(ACTIVATIONS) + 1))
        # print(f"——> mutated activation function at index {mutation_point}: {activations[mutation_point]}")

    # Mutate dropout rates
    if np.random.rand() < mutation_probability:
        mutation_point = np.random.randint(0, num_layers)
        dropout_rate_old = dropout_rates[mutation_point]
        while dropout_rate_old == dropout_rates[mutation_point]:
            dropout_rates[mutation_point] = float(np.random.uniform(0.1, 0.5))
        # print(f"——> mutated dropout rate at index {mutation_point}: {dropout_rates[mutation_point]}")

    # Mutate batch normalization settings
    if np.random.rand() < mutation_probability:
        mutation_point = np.random.randint(0, num_layers)
        batch_norm_old = batch_norms[mutation_point]
        while batch_norm_old == batch_norms[mutation_point]:
            batch_norms[mutation_point] = float(np.random.choice([0, 1]))
        # print(f"——> mutated batch normalization setting at index {mutation_point}: {batch_norms[mutation_point]}")
            
    # Pad each part to MAX_LAYERS length
    num_layers_pad = MAX_LAYERS - num_layers
        
    hidden_layer_sizes = np.append(hidden_layer_sizes, [0] * num_layers_pad)    # 0 indicates padding
    activations = np.append(activations, [-1] * num_layers_pad)                 # -1 indicates padding
    dropout_rates = np.append(dropout_rates, [-1.0] * num_layers_pad)           # -1 indicates padding
    batch_norms = np.append(batch_norms, [-1] * num_layers_pad)                 # -1 indicates padding
        
    individual = np.concatenate(([learning_rate, batch_size, epochs, patience, num_layers], hidden_layer_sizes, activations, dropout_rates, batch_norms, [individual[-1]]))   
    # print(f"——> after mutation:\n{individual}\n")

    return individual
    
def geneticAlgorithm():
    sol_per_pop = 15
    num_generations = 100
    num_parents_mating = 7
    K_tournaments = 5
    keep_parents = -1
    
    print("Parameters of the genetic algorithm:")
    print(f"—> Number of solutions in the population: {sol_per_pop}")
    print(f"—> Number of generations: {num_generations}")
    print(f"—> Number of parents mating: {num_parents_mating}\n")
    print(f"—> Number of individuals participating in each tournament: {K_tournaments}")
    print(f"—> Number of parents to keep: {keep_parents}\n")
    print(f"—> Maximum number of layers: {MAX_LAYERS}")
    print(f"—> Maximum number of neurons per layer: {MAX_LAYER_SIZE}\n")
    
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

    # Run the genetic algorithm
    ga_instance.run()
    
    # Plot the fitness history
    ga_instance.plot_fitness()
    
    return ga_instance

if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=2)
    
    # ——————————— CLASSIFICATION DATASETS ———————————
    if dataset == 'iris':
        # Load the Iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target

        # One-hot encode the labels
        y = to_categorical(y)

        # Split the dataset into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
    elif dataset == 'mnist':
        # Load the MNIST dataset
        (X_train, y_train), (X_test, y_test) = mnist.load_data(test_size=test_size)

        # Preprocess the data
        X_train = X_train.reshape(60000, 784).astype("float32") / 255
        X_test = X_test.reshape(10000, 784).astype("float32") / 255

        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)
    
    # ——————————— REGRESSION DATASETS ———————————
    elif dataset == 'california':
        # Load the California Housing dataset
        california = fetch_california_housing()
        X = california.data
        y = california.target

        # Normalize the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split the dataset into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    elif dataset == 'diabetes':
        # Load the Diabetes dataset
        diabetes = load_diabetes()
        X = diabetes.data
        y = diabetes.target

        # Normalize the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split the dataset into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    
    for i, penalty_mult in enumerate(penalty_mult_list):
        sys.stdout = open(f'./logs/{problem_type}/{dataset}/{i+1}.txt', 'w')
        print(f"Dataset: {dataset}")
        print(f"Number of training samples: {X_train.shape[0]}")
        print(f"Number of test samples: {X_test.shape[0]}\n")
        
        start = time.time()
        ga_instance = geneticAlgorithm()
        
        filename = f'genetic{i}'
        ga_instance.save(filename=filename)
        
        # Convert end time to hours, minutes, and seconds
        end = time.time() - start
        hours, rem = divmod(end, 3600)
        minutes, seconds = divmod(rem, 60)

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        best_solution = solution
        
        nn2 = array_to_nn(best_solution)

        # Training and evaluating nn2 on dataset
        nn2.model.fit(X_train, y_train, 
                        epochs=int(nn2.epochs), 
                        batch_size=int(nn2.batch_size), 
                        verbose=0, 
                        validation_data=(X_test, y_test),
                        callbacks=[nn2.early_stopping])

        # Summary of the model
        nn2.model.summary()
        
        # Evaluate the model on the training data
        train_loss, train_accuracy = nn2.model.evaluate(X_train, y_train)
        print('—> Training accuracy:', train_accuracy)

        # Evaluate the model on the test data
        test_loss, test_accuracy = nn2.model.evaluate(X_test, y_test)
        print('—> Test accuracy:', test_accuracy)
        
        # ga_instance.plot_fitness()

        print(f"Parameters of the best solution : {solution}")
        print(f"Fitness value of the best solution = {solution_fitness}")
        print(f"Index of the best solution : {solution_idx}")
        print(f"Validation accuracy of the best solution = {test_accuracy}")
        print(f"Validation loss of the best solution = {test_loss}")
        print(f"Elapsed time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
        
        plt.figure()
        plt.plot(fitness_history_best)
        plt.savefig(f"./logs/{problem_type}/{dataset}/plots/{i+1}max.jpg")
        plt.show(block=False)

        plt.figure()
        plt.plot(fitness_history_best)
        plt.savefig(f"./logs/{problem_type}/{dataset}/plots/{i+1}avg.jpg")
        plt.show(block=False)