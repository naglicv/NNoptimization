from model import NeuralNetwork

BEG_PARAMS = ("learning_rate", "batch_size", "epochs", "patience")

def array_to_nn(ga_array, input_size, output_size, problem_type, MAX_LAYERS, ACTIVATIONS, ACTIVATIONS_OUTPUT, device):
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

    nn = NeuralNetwork(input_size=input_size,
                       output_size=output_size,
                       problem_type=problem_type,
                       learning_rate=learning_rate,
                       batch_size=batch_size,
                       epochs=epochs,
                       patience=patience,
                       num_layers=num_layers,
                       hidden_layer_sizes=list(map(int, hidden_layer_sizes)),
                       activations=activations,
                       dropout_rates=list(map(float, dropout_rates)),
                       batch_norms=list(map(int, batch_norms)),
                       activation_output=activation_output,
                       device=device).to(device) 

    return nn
