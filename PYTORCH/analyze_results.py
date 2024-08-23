import os
import pandas as pd
import matplotlib.pyplot as plt

penalty_mult_list = [0, 0.01, 0.1, 1, 10, 100]  # Penalty multiplier for the complexity of the network


def plot_fitness_and_loss(data_path, dataset_type):
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']  # Colors for consistency

    # Go through all folders in the given path
    for dataset in os.listdir(data_path):
        dataset_path = os.path.join(data_path, dataset)
        
        if os.path.isdir(dataset_path):
            plt.figure(figsize=(18, 12))

            # Plotting 6 separate graphs for Max Fitness Score
            for i in range(1, len(penalty_mult_list) + 1):  # For each penalty multiplier
                plt.subplot(3, 2, i)
                file_path = os.path.join(dataset_path, f'{i}_fitness_history.csv')
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    plt.plot(df['Generation'], df['Max Fitness'], color=colors[i-1])
                    plt.title(f'Evolucija maksimalne ocene uspešnosti - {dataset} ({dataset_type}) - penalizacijski faktor = {penalty_mult_list[i-1]}')
                    plt.xlabel('Generacija')
                    plt.ylabel('Maksimalna ocena uspešnosti')
                    plt.grid(True)
                    # plt.legend()

            # Save the figure with Max Fitness graphs
            plt.tight_layout()
            plt.savefig(os.path.join(dataset_path, f'{dataset}_evolucija_maksimalne_uspesnosti.png'))
            plt.show()
            plt.close()

            # Plotting one graph for Validation Loss
            plt.figure(figsize=(10, 6))
            for i in range(1, 7):  # For each penalty multiplier
                file_path = os.path.join(dataset_path, f'{i}_fitness_history.csv')
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    plt.plot(df['Generation'], df['Chosen Individual Loss'], label=f'penalizacijski faktor = {penalty_mult_list[i-1]}', color=colors[i-1])
            plt.title(f'Evolucija validacijske izgube - {dataset} ({dataset_type})')
            plt.xlabel('Generacija')
            plt.ylabel('Validacijska izguba')
            plt.legend()
            plt.grid(True)

            # Save the figure with Validation Loss graph
            plt.tight_layout()
            plt.savefig(os.path.join(dataset_path, f'{dataset}_evolucija_validacijske_izgube.png'))
            plt.show()
            plt.close()

# Paths to directories
classification_path = 'logs/classification'
regression_path = 'logs/regression'

# Plot graphs for classification datasets
plot_fitness_and_loss(classification_path, 'klasifikacija')

# Plot graphs for regression datasets
plot_fitness_and_loss(regression_path, 'regresija')
