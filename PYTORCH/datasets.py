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
    'abalone'   : 1,                   # Medium, 8 features
    'parkinsons_telemonitoring': 189,  # Medium, 26 features (time series data)
    'solar_flare': 89,              # Medium, 10 features 
    'bike_sharing': 275,            # Medium, 16 features (mixed)
    'infrared_thermography_temperature': 925,  # Large, 7 features (time series data)
}

DATASET_LIST_SMALL = {
    # Classification Datasets
    'iris': 53,                     # Small, 4 features
    'wine': 109,                    # Small, 13 features
    'auto_mpg': 9,                  # Small, 7 features (mixed)
    'energy_efficiency': 242,       # Small, 8 features
    'breast_cancer_wisconsin_original': 15,  # Small, 9 features
    'heart_disease': 45,            # Medium, 13 features (mixed)
    'concrete_compressive_strength': 165,  # Medium, 8 features
    'abalone': 1,                   # Medium, 8 features
}

DATASET_LIST_LARGE = {
    # Regression Datasets
    'bank_marketing': 222,          # Medium, 16 features (mixed)
    'solar_flare': 89,              # Medium, 10 features 
    'parkinsons_telemonitoring': 189,  # Medium, 26 features (time series data)
    'bike_sharing': 275,            # Medium, 16 features (mixed)
    'adult': 2,                     # Large, 14 features
    'dry_bean': 602,                # Large, 16 features
    'infrared_thermography_temperature': 925,  # Large, 7 features (time series data)
    'optical_recognition_of_handwritten_digits': 80,  # Large, 64 features (image data)
}

def load_and_define_parameters(dataset):
    global problem_type, MAX_LAYERS, MAX_LAYER_SIZE, INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, ACTIVATIONS, ACTIVATIONS_OUTPUT

    if dataset in DATASET_LIST_CLASS:
        problem_type = "classification"
    elif dataset in DATASET_LIST_REG:
        problem_type = "regression"
    else:
        raise ValueError(f"Dataset '{dataset}' is not defined.")

    MAX_LAYERS = 20
    MAX_LAYER_SIZE = 512

    # Define activation functions based on the problem type
    ACTIVATIONS = {1: 'relu', 2: 'sigmoid', 3: 'tanh', 4: 'linear'}
    if problem_type == "classification":
        ACTIVATIONS_OUTPUT = {1: 'softmax'}
    elif problem_type == "regression":
        ACTIVATIONS_OUTPUT = {1: 'relu', 2: 'sigmoid', 3: 'tanh', 4: 'linear'}

    return problem_type, MAX_LAYERS, MAX_LAYER_SIZE, ACTIVATIONS, ACTIVATIONS_OUTPUT
