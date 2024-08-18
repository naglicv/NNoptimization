import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo  # UCI datasets
import pandas as pd

def load_and_preprocess_classification_data(dataset_name, dataset_id, device):
    # Fetch the dataset using the given ID
    data = fetch_ucirepo(id=dataset_id)
    
    # Extract features and labels
    X = data.data.features
    y = data.data.targets.squeeze()  # Ensure y is a Series by removing unnecessary dimensions
    
    # print(f"Dataset: {dataset_name}")
    # print(f"X shape: {X.shape}, y shape: {y.shape}")
    # print(f"Metadata:\n{data.metadata}")
    # print(f"Variables: {data.variables}")
    
    # Handle dataset-specific preprocessing
    if dataset_id == 53:  # Iris dataset
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
    elif dataset_id == 109:  # Wine dataset
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
    elif dataset_id == 2:  # Adult dataset
        # Preprocessing for numerical and categorical columns
        num_features = X.select_dtypes(include=['int64', 'float64']).columns
        cat_features = X.select_dtypes(include=['object']).columns
        num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features), ('cat', cat_transformer, cat_features)])
        X_preprocessed = preprocessor.fit_transform(X)
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        X_scaled = X_preprocessed
        
    elif dataset_id == 15:  # Breast Cancer Wisconsin dataset
        num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        X_preprocessed = num_transformer.fit_transform(X)
        y_encoded = y.replace({2: 0, 4: 1}).values
        X_scaled = X_preprocessed
        
    elif dataset_id == 45:  # Heart Disease dataset
        num_features = X.select_dtypes(include=['int64', 'float64']).columns
        cat_features = X.select_dtypes(include=['object', 'category']).columns
        num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features), ('cat', cat_transformer, cat_features)])
        X_preprocessed = preprocessor.fit_transform(X)
        y_encoded = y.apply(lambda x: 1 if x > 0 else 0).values
        X_scaled = X_preprocessed
        
    elif dataset_id == 222:  # Bank Marketing dataset
        num_features = X.select_dtypes(include=['int64', 'float64']).columns
        cat_features = X.select_dtypes(include=(['object', 'category'])).columns
        num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features), ('cat', cat_transformer, cat_features)])
        X_preprocessed = preprocessor.fit_transform(X)
        y_encoded = y.replace({"yes": 1, "no": 0}).values
        X_scaled = X_preprocessed

    elif dataset_id == 602:  # Dry Bean dataset
        num_features = X.select_dtypes(include=['int64', 'float64']).columns
        num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        cat_features = X.select_dtypes(include=['object']).columns
        cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features), ('cat', cat_transformer, cat_features)])
        X_preprocessed = preprocessor.fit_transform(X)
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        X_scaled = X_preprocessed

    elif dataset_id == 80:  # Optical Recognition of Handwritten Digits dataset
        num_features = X.select_dtypes(include=['int64', 'float64']).columns
        num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features)])
        X_preprocessed = preprocessor.fit_transform(X)
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        X_scaled = X_preprocessed

    else:
        raise ValueError("Dataset ID not recognized. Please update the function to handle this dataset.")
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long).to(device)
    
    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Determine input and output sizes
    input_size = X_train.shape[1]
    output_size = len(torch.unique(y_train))

    return input_size, output_size, X_train, y_train, X_val, y_val, X_test, y_test


def load_and_preprocess_regression_data(dataset_name, dataset_id, device):
    # Fetch the dataset using the given ID
    data = fetch_ucirepo(id=dataset_id)
    
    # Extract features and ensure the target columns are correctly loaded
    X = data.data.features
    y = data.data.targets

    # # Check if y is loaded correctly
    # if y is None:
    #     raise ValueError("Target data (y) is None. Ensure that the target columns are correctly loaded from the dataset.")
    
    # For multiple target columns, ensure y is a DataFrame and doesn't need squeezing
    if isinstance(y, pd.DataFrame):
        y = y  # Keep as DataFrame if there are multiple targets
    else:
        y = y.squeeze()  # Squeeze if it's a single target column

    print(f"Dataset: {dataset_name}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Metadata:\n{data.metadata}")
    print(f"Variables: {data.variables}")
    
    # Handle dataset-specific preprocessing
    if dataset_id == 9:  # Auto MPG dataset
        # Identify numerical and categorical features
        num_features = X.select_dtypes(include=['int64', 'float64']).columns
        cat_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Define preprocessing pipelines
        num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), 
                                          ('scaler', StandardScaler())])
        
        cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        
        # Combine transformers into a ColumnTransformer
        preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features), 
                                                       ('cat', cat_transformer, cat_features)])
        
        # Apply preprocessing
        X_preprocessed = preprocessor.fit_transform(X)
        X_scaled = X_preprocessed

        y = y.to_numpy().squeeze()  # Convert the DataFrame to a NumPy array and squeeze
        
    elif dataset_id == 242:  # Energy Efficiency dataset
        # Identify numerical and categorical features
        num_features = X.select_dtypes(include=['int64', 'float64']).columns
        cat_features = X.select_dtypes(include=['object', 'category']).columns  # None in this dataset
        
        # Define preprocessing pipelines
        num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        
        # Since there are no categorical features, no need for a categorical transformer
        preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features)])
        
        # Apply preprocessing
        X_preprocessed = preprocessor.fit_transform(X)
        X_scaled = X_preprocessed
        
        # Convert target to a NumPy array before converting to a tensor
        y = y.to_numpy()  # Convert the DataFrame to a NumPy array

    elif dataset_id == 165:  # Concrete Compressive Strength dataset
        # All features are numerical (continuous or integer)
        num_features = X.select_dtypes(include=['int64', 'float64']).columns
        
        # Define preprocessing pipelines for numerical data
        num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        
        # Since there are no categorical features, apply the preprocessor only to numerical features
        preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features)])
        
        # Apply preprocessing
        X_preprocessed = preprocessor.fit_transform(X)
        X_scaled = X_preprocessed
        
        # Convert target to a NumPy array before converting to a tensor
        y = y.to_numpy()  # Convert the DataFrame to a NumPy array
        
    elif dataset_id == 1:  # Abalone dataset
        # Identify numerical and categorical features
        num_features = X.select_dtypes(include=['int64', 'float64']).columns
        cat_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Define preprocessing pipelines
        num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        
        cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        
        # Combine transformers into a ColumnTransformer
        preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features),
                                                       ('cat', cat_transformer, cat_features)])
        
        # Apply preprocessing
        X_preprocessed = preprocessor.fit_transform(X)
        X_scaled = X_preprocessed
        
        # Convert target to a NumPy array before converting to a tensor
        y = y.to_numpy() 
        
    elif dataset_id == 189:  # Parkinsons Telemonitoring dataset
        # Identify numerical and categorical features
        num_features = X.select_dtypes(include=['int64', 'float64']).columns
        cat_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Define preprocessing pipelines
        num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        
        # Combine transformers into a ColumnTransformer
        preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features),
                                                       ('cat', cat_transformer, cat_features)])
        
        # Apply preprocessing
        X_preprocessed = preprocessor.fit_transform(X)
        X_scaled = X_preprocessed
        
        # Convert target to a NumPy array
        y = y.to_numpy()
    
    elif dataset_id == 89:  # Solar Flare dataset
        # All features are categorical in this dataset
        cat_features = X.columns
        
        # Define preprocessing pipeline for categorical data
        cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        
        # Apply the transformer to all categorical features
        preprocessor = ColumnTransformer(transformers=[('cat', cat_transformer, cat_features)])
        
        # Apply preprocessing
        X_preprocessed = preprocessor.fit_transform(X)
        X_scaled = X_preprocessed
        
        y = y.to_numpy()  # Convert the DataFrame to a NumPy array

    elif dataset_id == 275:  # Bike Sharing dataset
        # Separate the categorical and numerical features
        cat_features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
        num_features = ['temp', 'atemp', 'hum', 'windspeed']

        # Define preprocessing pipelines for categorical and numerical data
        cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        num_transformer = Pipeline(steps=[('scaler', StandardScaler())])

        # Apply the transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_features),
                ('cat', cat_transformer, cat_features)
            ]
        )

        # Apply preprocessing to the features
        X_preprocessed = preprocessor.fit_transform(X)
        X_scaled = X_preprocessed
        
        # Convert the target DataFrame to a NumPy array 
        y = y.to_numpy()

    elif dataset_id == 925:  # Infrared Thermography Temperature dataset
        # Separate the categorical and numerical features
        cat_features = ['Gender', 'Age', 'Ethnicity']
        num_features = [col for col in X.columns if col not in cat_features]

        # Define preprocessing pipelines for categorical and numerical data
        cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        num_transformer = Pipeline(steps=[('scaler', StandardScaler())])

        # Apply the transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_features),
                ('cat', cat_transformer, cat_features)
            ]
        )

        # Apply preprocessing to the features
        X_preprocessed = preprocessor.fit_transform(X)
        X_scaled = X_preprocessed
        
        # Convert the target DataFrame to a NumPy array and then to a PyTorch tensor
        y = y.to_numpy()

    else:
        raise ValueError("Dataset ID not recognized. Please update the function to handle this dataset.")
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    
    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Determine input size
    input_size = X_train.shape[1]
    output_size = 1

    return input_size, output_size, X_train, y_train, X_val, y_val, X_test, y_test