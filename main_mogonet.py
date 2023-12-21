import torch
import os
import time
import random
from train_test import train_test

# Function to save the model components
def save_model(model_dict, model_folder, identifier):
    os.makedirs(model_folder, exist_ok=True)  # Ensure the directory exists
    for key, model in model_dict.items():
        filename = f"{key}_best_{identifier}.pth"
        file_path = os.path.join(model_folder, filename)
        torch.save(model.state_dict(), file_path)

# Function to randomly select hyperparameters
def random_hyperparameters():
    return {
        'lr_e_pretrain': random.choice([1e-4, 1e-3, 5e-3, 1e-2, 5e-2]),
        'lr_e': random.choice([1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
        'lr_c': random.choice([1e-4, 1e-3, 5e-3, 1e-2, 5e-2]),
        'num_epoch_pretrain': random.choice([100, 300, 500, 700, 900]),
        'num_epoch': random.choice([1000, 1500, 2000, 2500, 3000]),
        'num_data_points': None  # Assumes you want to use all data points
    }

# Main execution block
if __name__ == "__main__":
    data_folder = 'BRCA'
    view_list = [1, 2, 3]
    num_class = 5 if data_folder == 'BRCA' else 2
    model_folder = os.path.join(data_folder, 'models')

    cuda = torch.cuda.is_available()
    num_runs = 1  # Number of random searches
    start_time = time.time()
    for _ in range(num_runs):
        hyperparameters = random_hyperparameters()

        # Run train_test with randomly selected hyperparameters
        #start_time = time.time()
        
        model_dict, final_result = train_test(
            data_folder, view_list, num_class,
            hyperparameters['lr_e_pretrain'],
            hyperparameters['lr_e'],
            hyperparameters['lr_c'],
            hyperparameters['num_epoch_pretrain'],
            hyperparameters['num_epoch'],
            hyperparameters['num_data_points']
        )

    end_time = time.time()
    duration = end_time - start_time

    # Save the model components
    identifier = int(time.time())  # Use current timestamp as a unique identifier
    save_model(model_dict, model_folder, identifier)

    print(f"Run completed in {duration} seconds. Model saved with identifier {identifier}.")
