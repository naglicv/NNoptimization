import pygad
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from nn2 import NeuralNetwork, array_to_nn, plot_data, LEARNING_RATE, MAX_LAYERS, MAX_LAYER_SIZE, INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, ACTIVATIONS, ACTIVATIONS_OUTPUT
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys


fitness_history = []
gen_num = 0

def generatePopulation(sol_per_pop):
    population = np.array([])
    for _ in range(sol_per_pop):
        # Generate random solution parts
        num_layers = np.random.randint(1, MAX_LAYERS + 1)
        hidden_layer_sizes = [np.random.randint(1, MAX_LAYER_SIZE + 1) for _ in range(num_layers)]
        activations = [np.random.randint(1, len(ACTIVATIONS) + 1) for _ in range(num_layers)]
        dropout_rates = [np.random.uniform(0.1, 0.5) for _ in range(num_layers)]
        batch_norms = [np.random.choice([0, 1]) for _ in range(num_layers)]
        activation_output = np.random.randint(1, len(ACTIVATIONS_OUTPUT) + 1)

        # Pad each part to MAX_LAYERS length
        hidden_layer_sizes += [0] * (MAX_LAYERS - num_layers)  # 0 indicates padding
        activations += [-1] * (MAX_LAYERS - num_layers)  #-1 indicates padding
        dropout_rates += [-1.0] * (MAX_LAYERS - num_layers)  # -1 indicates padding
        batch_norms += [-1] * (MAX_LAYERS - num_layers)  # -1 indicates padding

        # Combine all parts into a single solution array
        solution = np.array([num_layers] + hidden_layer_sizes + activations + dropout_rates + batch_norms + [activation_output])
        
        # Check if population is empty
        if population.size == 0:
            # Initialize population with solution as the first nested array
            population = np.array([solution])
        else:
            # Concatenate solution as a new row to the existing population
            population = np.vstack((population, solution))
        
    print("————————————————————> POPULATION: ", population)
    return population

def fitness_func(ga_instance, solution, solution_idx):
    # Create a neural network from the solution array
    solution_nn = array_to_nn(solution)
    # Train the neural network
    history = solution_nn.model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_data=(X_test, y_test))
    # Get the validation accuracy
    validation_accuracy = history.history['val_accuracy'][-1]
    
    # Calculate the number of layers and total number of neurons
    num_layers = solution[0]
    total_neurons = np.sum(solution[1:MAX_LAYERS + 1])
    
    # Calculate the relative complexity of the network
    relative_layers = num_layers / MAX_LAYERS
    relative_neurons = total_neurons / (MAX_LAYERS * MAX_LAYER_SIZE)  # Maximum possible neurons
    
    # Calculate the penalty based on relative complexity
    penalty = relative_layers * 0.01 + relative_neurons * 0.001  # Adjust weights as needed
    
    # Calculate the fitness score
    fitness_score = validation_accuracy - penalty
    print("————————————————————> solution: ", solution)
    print(f"——————> fitness score = validation accuracy - penalty = {validation_accuracy} - {penalty} = {fitness_score}\n")
    return fitness_score

def custom_crossover(parents, offspring_size, ga_instance):
    offspring = []
    print("————————————————————> PARENTS: ", parents)
    print("————————————————————> OFFSPRING SIZE: ", offspring_size)
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
    # Parameters of parent 1
    num_layers1 = int(parent1[0])
    hidden_layer_sizes1 = parent1[1:num_layers1 + 1].astype(np.int32)
    activations1 = parent1[MAX_LAYERS + 1:MAX_LAYERS + num_layers1 + 1].astype(np.int32)
    dropout_rates1 = parent1[2 * MAX_LAYERS + 1:2 * MAX_LAYERS + num_layers1 + 1]
    batch_norms1 = parent1[3 * MAX_LAYERS + 1:3 * MAX_LAYERS + num_layers1 + 1].astype(np.int32)
    
    # Parameters of parent 2
    num_layers2 = int(parent2[0])
    hidden_layer_sizes2 = parent2[1:num_layers2 + 1].astype(np.int32)
    activations2 = parent2[MAX_LAYERS + 1:MAX_LAYERS + num_layers2 + 1].astype(np.int32)
    dropout_rates2 = parent2[2 * MAX_LAYERS + 1:2 * MAX_LAYERS + num_layers2 + 1]
    batch_norms2 = parent2[3 * MAX_LAYERS + 1:3 * MAX_LAYERS + num_layers2 + 1].astype(np.int32)
    
    print("parent1: ", parent1)
    print("parent2: ", parent2)
    
    # Select crossover points
    cross_len = np.min([num_layers1, num_layers2]) - 1
    
    parent1_point1 = np.random.randint(0, num_layers1 - cross_len)
    parent1_point2 = parent1_point1 + cross_len
    
    parent2_point1 = np.random.randint(0, num_layers2 - cross_len)
    parent2_point2 = parent2_point1 + cross_len
    
    # Perform crossover
    new_hidden_layer_sizes1 = np.concatenate((hidden_layer_sizes1[:parent1_point1], hidden_layer_sizes2[parent2_point1:parent2_point2], hidden_layer_sizes1[parent1_point2:]))
    new_hidden_layer_sizes2 = np.concatenate((hidden_layer_sizes2[:parent2_point1], hidden_layer_sizes1[parent1_point1:parent1_point2], hidden_layer_sizes2[parent2_point2:]))
    
    new_num_layers1 = len(new_hidden_layer_sizes1)
    new_num_layers2 = len(new_hidden_layer_sizes2)
    
    new_activations1 = np.concatenate((activations1[:parent1_point1], activations2[parent2_point1:parent2_point2], activations1[parent1_point2:]))
    new_activations2 = np.concatenate((activations2[:parent2_point1], activations1[parent1_point1:parent1_point2], activations2[parent2_point2:]))
    
    new_dropout_rates1 = np.concatenate((dropout_rates1[:parent1_point1], dropout_rates2[parent2_point1:parent2_point2], dropout_rates1[parent1_point2:]))
    new_dropout_rates2 = np.concatenate((dropout_rates2[:parent2_point1], dropout_rates1[parent1_point1:parent1_point2], dropout_rates2[parent2_point2:]))
    
    new_batch_norms1 = np.concatenate((batch_norms1[:parent1_point1], batch_norms2[parent2_point1:parent2_point2], batch_norms1[parent1_point2:]))
    new_batch_norms2 = np.concatenate((batch_norms2[:parent2_point1], batch_norms1[parent1_point1:parent1_point2], batch_norms2[parent2_point2:]))

    # print("————————————————————> NEW HIDDEN LAYER SIZES 1: ", new_hidden_layer_sizes1)
    # print("————————————————————> NEW HIDDEN LAYER SIZES 2: ", new_hidden_layer_sizes2)
    # print("————————————————————> NEW ACTIVATIONS 1: ", new_activations1)
    # print("————————————————————> NEW ACTIVATIONS 2: ", new_activations2)
    # print("————————————————————> NEW DROPOUT RATES 1: ", new_dropout_rates1)
    # print("————————————————————> NEW DROPOUT RATES 2: ", new_dropout_rates2)
    # print("————————————————————> NEW BATCH NORMS 1: ", new_batch_norms1)
    # print("————————————————————> NEW BATCH NORMS 2: ", new_batch_norms2)

    # Pad each part to MAX_LAYERS length
    new_hidden_layer_sizes1 = np.append(new_hidden_layer_sizes1, [0] * (MAX_LAYERS - new_num_layers1))  # 0 indicates padding
    new_hidden_layer_sizes2 = np.append(new_hidden_layer_sizes2, [0] * (MAX_LAYERS - new_num_layers2))
    new_activations1 = np.append(new_activations1, [-1] * (MAX_LAYERS - new_num_layers1))               # -1 indicates padding
    new_activations2 = np.append(new_activations2, [-1] * (MAX_LAYERS - new_num_layers2))
    new_dropout_rates1 = np.append(new_dropout_rates1, [-1.0] * (MAX_LAYERS - new_num_layers1))         # -1 indicates padding
    new_dropout_rates2 = np.append(new_dropout_rates2, [-1.0] * (MAX_LAYERS - new_num_layers2))
    new_batch_norms1 = np.append(new_batch_norms1, [-1] * (MAX_LAYERS - new_num_layers1))               # -1 indicates padding
    new_batch_norms2 = np.append(new_batch_norms2, [-1] * (MAX_LAYERS - new_num_layers2))
        
    offspring1 = np.concatenate([
        [new_num_layers1],
        new_hidden_layer_sizes1.astype(np.float32), 
        new_activations1.astype(np.float32), 
        new_dropout_rates1.astype(np.float32), 
        new_batch_norms1.astype(np.float32), 
        [parent1[-1] if np.random.rand() <= 0.5 else parent2[-1]]
    ])
    
    offspring2 = np.concatenate([
        [new_num_layers2],
        new_hidden_layer_sizes2.astype(np.float32),
        new_activations2.astype(np.float32),
        new_dropout_rates2.astype(np.float32),
        new_batch_norms2.astype(np.float32),
        [parent1[-1] if np.random.rand() <= 0.5 else parent2[-1]]
    ])

    return offspring1, offspring2

def custom_mutation(offspring, ga_instance):
    for i in range(len(offspring)):
        offspring[i] = structured_mutation(offspring[i])
    return np.array(offspring)

def structured_mutation(individual):
    print(f"————————————————————> INDIVIDUAL:\n{individual}\n")
    print("-> MUTATION\n")
    mutation_probability = 0.02
    
    # Parameters of individual
    num_layers_old = int(individual[0])
    hidden_layer_sizes_old = individual[1:MAX_LAYERS + 1]
    activations_old = individual[MAX_LAYERS + 1:2 * MAX_LAYERS + 1]
    dropout_rates_old = individual[2 * MAX_LAYERS + 1:3 * MAX_LAYERS + 1]
    batch_norms_old = individual[3 * MAX_LAYERS + 1:4 * MAX_LAYERS + 1]
    
    # Mutate number of layers
    if np.random.rand() < mutation_probability:
        while True:
            # Generate num_layers_new with a triangular distribution
            """Triangular Distribution Sampling: 
            This distribution is defined by three parameters: the minimum value (left), the mode 
            (mode), and the maximum value (right). 

            1 is the minimum value, num_layers_old (the current number of layers) is the mode 
            and MAX_LAYERS (a predefined constant representing the maximum possible number of layers)
            is the maximum value. 
            The np.random.triangular function from NumPy's random module is used to sample a value 
            based on this distribution, and the round function is applied to ensure the result is an 
            integer. This step aims to generate a new layer count that is likely close to the current 
            count but allows for variability within specified bounds"""
            num_layers_new = round(np.random.triangular(1, num_layers_old, MAX_LAYERS))

            # Ensure num_layers_new is within the bounds [1, MAX_LAYERS]
            """The min function first ensures that num_layers_new does not exceed MAX_LAYERS, and then 
            the max function ensures that the result is at least 1."""
            num_layers_new = max(1, min(num_layers_new, MAX_LAYERS))
            
            # Check if num_layers_new is different from num_layers_old
            if num_layers_new != num_layers_old:
                break
        
        print(f"number of layers: {num_layers_old} -> {num_layers_new}")
        individual[0] = num_layers_new

        if num_layers_new > num_layers_old:
            # Add new layers
            for i in range(num_layers_old, num_layers_new):
                individual[1 + i] = np.random.randint(1, MAX_LAYER_SIZE + 1)                 # Random hidden layer size
                individual[MAX_LAYERS + 1 + i] = np.random.randint(1, len(ACTIVATIONS) + 1)  # Random activation function
                individual[2 * MAX_LAYERS + 1 + i] = np.random.uniform(0.1, 0.5)             # Random dropout rate
                individual[3 * MAX_LAYERS + 1 + i] = np.random.choice([0, 1])                # Random batch normalization setting
        elif num_layers_new < num_layers_old:
            # Select layers to remove randomly
            layers_to_remove = np.random.choice(range(num_layers_old), num_layers_old - num_layers_new, replace=False)
            for i in layers_to_remove:
                individual[1 + i] = 0                       # Zero hidden layer size
                individual[MAX_LAYERS + 1 + i] = -1         # Indicate no activation function
                individual[2 * MAX_LAYERS + 1 + i] = -1.0   # Indicate no dropout
                individual[3 * MAX_LAYERS + 1 + i] = -1     # Indicate no batch normalization
            
            # Remove padding
            hidden_layer_sizes = individual[1:MAX_LAYERS + 1]
            mask = np.where(hidden_layer_sizes != 0)
            
            hidden_layer_sizes_new = hidden_layer_sizes[mask]
            activations_new = activations_old[mask]
            dropout_rates_new = dropout_rates_old[mask]
            batch_norms_new = batch_norms_old[mask]
            
            # Pad each part to MAX_LAYERS length
            num_layers_pad = MAX_LAYERS - num_layers_new
            
            new_hidden_layer_sizes = np.append(hidden_layer_sizes_new, [0] * num_layers_pad)    # 0 indicates padding
            new_activations = np.append(activations_new, [-1] * num_layers_pad)                 # -1 indicates padding
            new_dropout_rates = np.append(dropout_rates_new, [-1.0] * num_layers_pad)           # -1 indicates padding
            new_batch_norms = np.append(batch_norms_new, [-1] * num_layers_pad)                 # -1 indicates padding

            individual = np.array([num_layers_new] + new_hidden_layer_sizes + new_activations + new_dropout_rates + new_batch_norms + [individual[-1]])   
    else:
        num_layers_new = num_layers_old
        
    # Mutate hidden layer sizes
    for i in range(num_layers_new):
        if np.random.rand() < mutation_probability:
            individual[1 + i] = float(np.random.randint(1, MAX_LAYER_SIZE + 1))

    # Mutate activation functions
    for i in range(num_layers_new):
        if np.random.rand() < mutation_probability:
            individual[MAX_LAYERS + 1 + i] = float(np.random.randint(1, len(ACTIVATIONS) + 1))

    # Mutate dropout rates
    for i in range(num_layers_new):
        if np.random.rand() < mutation_probability:
            individual[2 * MAX_LAYERS + 1 + i] = float(np.random.uniform(0.1, 0.5))

    # Mutate batch normalization settings
    for i in range(num_layers_new):
        if np.random.rand() < mutation_probability:
            individual[3 * MAX_LAYERS + 1 + i] = float(np.random.choice([0, 1]))

    # Mutate output activation function
    if np.random.rand() < mutation_probability:
        individual[-1] = float(np.random.randint(1, len(ACTIVATIONS_OUTPUT) + 1))

    return individual

def update_plot(num, fitness_history, line):
    """
    Update function for FuncAnimation. Updates the plot with new data at each frame.
    """
    line.set_data(range(len(fitness_history[:num+1])), fitness_history[:num+1])
    gen_num += 1
    print(f"—————————————————————— GENERATION {gen_num} ————————————————————————")
    return line,
    
def geneticAlgorithm():
    sol_per_pop = 6
    num_generations = 8
    num_parents_mating = 3
    
    population = generatePopulation(sol_per_pop)
    
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           initial_population=population,
                           fitness_func=fitness_func,
                           parent_selection_type="tournament",
                           K_tournament=2,
                           keep_parents=-1,
                           crossover_type=custom_crossover,
                           mutation_type=custom_mutation,
                           save_best_solutions=True,
                           random_seed=42)

    # Run the genetic algorithm
    ga_instance.run()
    
    # Set up the animation
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0, len(fitness_history))
    ax.set_ylim(min(fitness_history), max(fitness_history))
    ax.set_title('Fitness over Generations')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    
    ani = animation.FuncAnimation(fig, update_plot, fargs=(fitness_history, line), frames=len(fitness_history), interval=200, repeat=False)
    
    # Display the animation
    plt.show()

    return ga_instance

if __name__ == '__main__':
    sys.stdout = open('logs/mn_6-8.txt', 'w')
    np.set_printoptions(suppress=True, precision=2)

    # Fix random seed for reproducibility
    # rnd_seed = 42
    # np.random.seed(rnd_seed)
    
    # # Load the Iris dataset
    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target

    # # plot_data(iris)

    # # One-hot encode the labels
    # y = to_categorical(y)

    # # Split the dataset into a training set and a test set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocess the data
    X_train = X_train.reshape(60000, 784).astype("float32") / 255
    X_test = X_test.reshape(10000, 784).astype("float32") / 255

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    
    ga_instance = geneticAlgorithm()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    best_solution = solution
    
    # Create a new NeuralNetwork instance using array_to_nn
    nn2 = array_to_nn(best_solution)
    
    filename = 'genetic'
    ga_instance.save(filename=filename)

    # Training and evaluating nn2 on dataset
    nn2.model.fit(X_train, y_train, epochs=100, batch_size=32)

    # Summary of the model
    nn2.summary()
    
    # Evaluate the model on the training data
    train_loss, train_accuracy = nn2.model.evaluate(X_train, y_train)
    print('—> Training accuracy:', train_accuracy)

    # Evaluate the model on the test data
    test_loss, test_accuracy = nn2.model.evaluate(X_test, y_test)
    print('—> Test accuracy:', test_accuracy)
    
    ga_instance.plot_fitness()

    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")
    