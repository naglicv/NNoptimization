import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo  # assuming this is a custom function for fetching UCI datasets

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
    
    # Determine input and output sizes
    input_size = X_train.shape[1]
    output_size = len(torch.unique(y_train))

    return input_size, output_size, X_train, y_train, X_val, y_val, X_test, y_test
