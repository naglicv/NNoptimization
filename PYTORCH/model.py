import torch
from torch import nn, optim


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, problem_type, learning_rate, batch_size, epochs, patience, num_layers, hidden_layer_sizes, activations, dropout_rates, batch_norms, activation_output, device):
        super(NeuralNetwork, self).__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.num_layers = num_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.problem_type = problem_type
        self.device = device

        # Define layer sizes, including input, hidden, and output layers
        layer_sizes = [input_size] + list(hidden_layer_sizes) + [output_size]
        self.layers = nn.ModuleList()

        # Loop to create the layers of the network
        for i in range(len(layer_sizes) - 1):
            if layer_sizes[i] == 0 or layer_sizes[i+1] == 0:
                raise ValueError(f"Layer size must not be zero: {layer_sizes[i]} -> {layer_sizes[i+1]}")
            # Add a fully connected (linear) layer
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            if i < num_layers:  # Apply to hidden layers only
                if batch_norms[i]:
                    self.layers.append(nn.BatchNorm1d(layer_sizes[i+1]))  # Add batch normalization if specified
                # Add activation function based on the configuration
                if activations[i] == 'relu':
                    self.layers.append(nn.ReLU())
                elif activations[i] == 'sigmoid':
                    self.layers.append(nn.Sigmoid())
                elif activations[i] == 'tanh':
                    self.layers.append(nn.Tanh())
                elif activations[i] == 'linear':
                    self.layers.append(nn.Identity())
                # Add dropout if specified
                if dropout_rates[i] > 0:
                    self.layers.append(nn.Dropout(dropout_rates[i]))

        # Set the output activation based on the problem type
        if problem_type == "classification":
            self.output_activation = nn.Identity()  # No activation needed, handled by CrossEntropyLoss
            self.criterion = nn.CrossEntropyLoss()
        elif problem_type == "regression":
            self.output_activation = nn.Identity()  # Linear activation (no activation)
            self.criterion = nn.MSELoss()  # Mean Squared Error Loss for regression

        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        # Pass input through the layers except the last one
        for layer in self.layers[:-1]:
            x = layer(x)
        # Apply the final layer and output activation
        x = self.layers[-1](x)
        x = self.output_activation(x)
        return x
    
    def train_model(self, train_loader, val_loader=None):
        self.train()  # Set the model to training mode
        best_val_loss = float('inf')
        epochs_no_improve = 0

        # Training loop
        for epoch in range(self.epochs):
            running_loss = 0.0
            for batch_data, batch_labels in train_loader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                if batch_data.size(0) == 1:
                    continue

                self.optimizer.zero_grad()  # Clear the gradients
                outputs = self(batch_data)
                
                # Adjust labels based on problem type
                if self.problem_type == "classification":
                    batch_labels = batch_labels.long()
                elif self.problem_type == "regression":
                    batch_labels = batch_labels.view(outputs.size())
                
                # Compute the loss and backpropagate
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # Validate model performance if a validation set is provided
            if val_loader:
                val_loss, val_accuracy = self.evaluate(val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                # Early stopping based on patience
                if epochs_no_improve >= self.patience:
                    break

    def evaluate(self, data_loader):
        self.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient calculation for evaluation
            for batch_data, batch_labels in data_loader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                outputs = self(batch_data)
                
                # Adjust labels based on problem type
                if self.problem_type == "classification":
                    batch_labels = batch_labels.long()
                    loss = self.criterion(outputs, batch_labels)
                    total_loss += loss.item()
                    
                    # Calculate accuracy for classification
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                elif self.problem_type == "regression":
                    batch_labels = batch_labels.view_as(outputs)
                    loss = self.criterion(outputs, batch_labels)
                    total_loss += loss.item()

        # Calculate average loss and accuracy
        average_loss = total_loss / len(data_loader)
        accuracy = (correct / total) * 100 if self.problem_type == "classification" else None
        return average_loss, accuracy
