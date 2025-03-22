import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from helpers import read_data_demo, plot_decision_boundaries
from torchvision.transforms import ToTensor
import pandas as pd


class EuropeDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
        """
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        self.samples = torch.tensor(read_data_demo(csv_file)[0][:,1:-1], dtype=torch.float32).to(device)
        self.labels = torch.tensor(read_data_demo(csv_file)[0][:,-1], dtype=torch.long).to(device)


        # Load the data into a tensors
        # The features shape is (n,d)
        # The labels shape is (n)
        # The feature dtype is float
        # THe labels dtype is long

        #### END OF YOUR CODE ####


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data row
        
        Returns:
            dictionary or list corresponding to a feature tensor and it's corresponding label tensor
        """
        label = self.labels[idx]
        sample = self.samples[idx]
        return sample, label


class MLP(nn.Module):
    def __init__(self, num_hidden_layers, hidden_dim, input_dim, output_dim):
        super(MLP, self).__init__()

        # Increase hidden_dim since we're mapping geographic coordinates to countries
        self.hidden_dim = hidden_dim

        # First layer:
        self.first_layer = nn.Linear(input_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Middle layers with their own batch norms
        self.mid_layers = nn.ModuleList()
        for _ in range(num_hidden_layers - 1):
            self.mid_layers.append(nn.Linear(hidden_dim, hidden_dim))
            # self.mid_layers.append(nn.BatchNorm1d(hidden_dim))

        # Output layer:
        self.last_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # First layer
        x = self.first_layer(x)
        x = self.relu(x)
        # if x.shape[1] == out.shape[1]:
        #     x = x + out
        # else:
        #     x = out

        # Only apply batch norm if batch size > 1
        # if x.size(0) > 1:
        #     x = self.bn1(x)


        # Middle layers
        for i in range(0, len(self.mid_layers)):
            x = self.mid_layers[i](x)
            x = self.relu(x)
            # x = out + identity # add residual connection

            # Conditional batch normalization
            # if x.size(0) > 1:
            #     x = self.mid_layers[i + 1](x)  # BatchNorm

        # Output layer
        x = self.last_layer(x)
        return x


def train(train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=50, batch_size=256):
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []
    iterations_num = []

    model.to(device)

    for ep in range(epochs):
        model.train()
        ep_train_acc = 0
        ep_train_loss = 0
        num_batches = 0

        for batch_idx, (samples, labels) in enumerate(trainloader):
            samples = samples.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(samples)
            loss = criterion(output, labels)
            train_accuracy = (output.argmax(dim=1) == labels).float().mean().item()
            loss.backward()
            optimizer.step()

            # Store individual batch loss
            ep_train_loss += loss.item()
            iterations_num.append(loss.item())
            ep_train_acc += train_accuracy
            num_batches += 1

        train_accs.append(ep_train_acc / num_batches)
        train_losses.append(ep_train_loss / num_batches)


        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            num_val_batches = 0

            for samples, labels in valloader:
                samples = samples.to(device)
                labels = labels.to(device)
                val_output = model(samples)
                loss = criterion(val_output, labels)
                acc = (val_output.argmax(dim=1) == labels).float().mean().item()
                val_loss += loss.item()
                val_acc += acc
                num_val_batches += 1

            val_losses.append(val_loss / num_val_batches)
            val_accs.append(val_acc / num_val_batches)

            # Test phase
            test_loss = 0
            test_acc = 0
            num_test_batches = 0

            for samples, labels in testloader:
                samples = samples.to(device)
                labels = labels.to(device)
                test_output = model(samples)
                test_loss += criterion(test_output, labels).item()
                test_acc += (test_output.argmax(dim=1) == labels).float().mean().item()
                num_test_batches += 1

            test_accs.append(test_acc / num_test_batches)
            test_losses.append(test_loss / num_test_batches)

        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(
            ep, train_accs[-1], val_accs[-1], test_accs[-1]))

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, iterations_num


def different_lr(train_dataset, val_dataset, test_dataset, epochs=50, batch_size=256):
    lrs = [1,0.01, 0.001, 0.00001]
    colors = ['red', 'blue', 'green', 'yellow']

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for lr_idx in range(len(lrs)):
        print(f"\n=== Starting experiment with lr={lrs[lr_idx]} ===")
        # Clear everything
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)

        # Reset all seeds
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        random.seed(0)



        # Create new model
        model = MLP(6, 16, input_dim, output_dim)

        # Explicitly reinitialize all parameters
        model = model.to(device)

        # Train with current learning rate
        model, _, _, _, _, val_losses, _,_ = train(
            train_dataset,
            val_dataset,
            test_dataset,
            model,
            lr=lrs[lr_idx],
            epochs=epochs,
            batch_size=batch_size
        )

        plt.plot(val_losses, label=f'lr = {lrs[lr_idx]}', color=colors[lr_idx])

    plt.title('Validation Losses for Different Learning Rates')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


def different_batch_size(train_dataset, val_dataset, test_dataset, lr=0.001):
    batches = [1, 16, 128, 1024]
    epochs = [1, 10, 50, 50]  # Different epochs for different batch sizes
    colors = ['red', 'blue', 'green', 'yellow']

    plt.figure(figsize=(15, 5))

    # First subplot: Validation Accuracy vs Epoch
    plt.subplot(1, 2, 1)
    all_val_accs = []  # Store validation accuracies for all batch sizes
    all_batch_losses = []  # Store training losses for all batch sizes

    for batch_idx, batch_size in enumerate(batches):
        # Reset seeds for reproducibility
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)

        current_epochs = epochs[batch_idx]

        # Create new model instance
        model = MLP(6, 16, input_dim, output_dim)

        # Train the model
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, iterations = train(
            train_dataset,
            val_dataset,
            test_dataset,
            model,
            lr=lr,
            epochs=current_epochs,
            batch_size=batch_size
        )

        all_val_accs.append(val_accs)
        all_batch_losses.append(iterations)

        # Plot validation accuracy vs epoch
        epochs_x = np.linspace(0, current_epochs, len(val_accs))
        plt.plot(epochs_x, val_accs, label=f'Batch Size = {batch_size}', color=colors[batch_idx])

    plt.title('Validation Accuracy vs Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Second subplot: Training Loss vs Batch
    plt.subplot(1, 2, 2)
    for batch_idx, batch_size in enumerate(batches):
        n_samples = len(train_dataset)
        batches_per_epoch = (n_samples + batch_size - 1) // batch_size
        total_batches = batches_per_epoch * epochs[batch_idx]

        # Get actual number of batches used
        batch_numbers = np.arange(len(all_batch_losses[batch_idx]))
        plt.plot(batch_numbers, all_batch_losses[batch_idx],
                 label=f'Batch Size = {batch_size}', color=colors[batch_idx])

    plt.title('Training Loss vs Batch')
    plt.xlabel('Batch Number (actual updates)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\nTraining Statistics:")
    for batch_idx, batch_size in enumerate(batches):
        n_samples = len(train_dataset)
        updates_per_epoch = (n_samples + batch_size - 1) // batch_size
        total_updates = updates_per_epoch * epochs[batch_idx]
        print(f"\nBatch size {batch_size}:")
        print(f"- Updates per epoch: {updates_per_epoch}")
        print(f"- Total updates: {total_updates}")
        print(f"- Final validation accuracy: {all_val_accs[batch_idx][-1]:.3f}")
def different_depth_and_width(train_dataset, val_dataset, test_dataset, lr=0.001):
    widths = [16,16,16,16,8,32,64]
    depths = [1,2,6,10,6,6,6]
    best_train_L = []
    best_val_L = []
    best_test_L = []
    worst_train_L = []
    worst_val_L = []
    worst_test_L = []
    best_acc = 0
    best_width = 0
    best_depth = 0
    worst_acc = 1
    worst_width = 0
    worst_depth = 0
    best_model = None
    worst_model =None
    for i in range(len(widths)):
        model = MLP(depths[i], widths[i], input_dim, output_dim)
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses,_ = train(train_dataset, val_dataset, test_dataset, model, lr=lr, epochs=50, batch_size=256)
        if val_accs[-1] > best_acc:
            best_model = model
            best_acc = val_accs[-1]
            best_width = widths[i]
            best_depth = depths[i]
            best_train_L = train_losses
            best_val_L = val_losses
            best_test_L = test_losses
        if val_accs[-1] < worst_acc:
            worst_model = model
            worst_acc = val_accs[-1]
            worst_width = widths[i]
            worst_depth = depths[i]
            worst_train_L = train_losses
            worst_val_L = val_losses
            worst_test_L = test_losses
    print(f"Best width: {best_width}, Best depth: {best_depth}, Best accuracy: {best_acc}")
    print(f"worst model width: {worst_width}, worst model depth: {worst_depth}, worst model accuracy: {worst_acc}")
    # Plot the losses of the best model
    plt.figure()
    plt.plot(best_train_L, label='Train', color='red')
    plt.plot(best_val_L, label='Val', color='blue')
    plt.plot(best_test_L, label='Test', color='green')
    plt.plot(worst_train_L, label='Train', color='red', linestyle='--')
    plt.plot(worst_val_L, label='Val', color='blue', linestyle='--')
    plt.plot(worst_test_L, label='Test', color='green', linestyle='--')
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')
    plot_decision_boundaries(best_model, test_data[['long', 'lat']].values, test_data['country'].values,
                             'Decision Boundaries', implicit_repr=False)
    plot_decision_boundaries(worst_model, test_data[['long', 'lat']].values, test_data['country'].values,
                             'Decision Boundaries', implicit_repr=False)


def default_run(model, train_dataset, val_dataset, test_dataset, lr=0.001, epochs=50, batch_size=256):
    # Training part
    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, _ = train(train_dataset, val_dataset,
                                                                                             test_dataset, model,
                                                                                             lr=lr, epochs=epochs,
                                                                                             batch_size=batch_size)
    # Loss plot
    plt.figure()
    plt.plot(train_losses, label='Train', color='red')
    plt.plot(val_losses, label='Val', color='blue')
    plt.plot(test_losses, label='Test', color='green')
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Accuracy plot
    plt.figure()
    plt.plot(train_accs, label='Train', color='red')
    plt.plot(val_accs, label='Val', color='blue')
    plt.plot(test_accs, label='Test', color='green')
    plt.title('Accs.')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Load data
    test_data = pd.read_csv('test.csv')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if train_dataset.samples.shape[1] == 20:
        test_features = transform_to_20D(test_data)
        # Get only the transformed feature columns for the first two alphas
        feature_cols = [col for col in test_features.columns if 'long_0.1' in col or 'long_0.2' in col]
        X = test_features[feature_cols].values
        y = test_data['country'].values

        # Plot decision boundaries
        plot_decision_boundaries(model, X, y, 'Decision Boundaries', implicit_repr=True)
    else:
        # Original 2D case
        X = test_data[['long', 'lat']].values
        y = test_data['country'].values
        plot_decision_boundaries(model, X, y, 'Decision Boundaries', implicit_repr=False)

def depth_of_network(width=16):
    depths = [1,2,6,10]
    accs = []
    for depth in depths:
        model = MLP(depth,width, input_dim, output_dim)
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, _ = train(train_dataset, val_dataset,
                                                                                                 test_dataset, model,
                                                                                                 lr=0.001, epochs=50,
                                                                                                 batch_size=256)
        accs.append(val_accs[-1])
    plt.figure()
    plt.plot(depths,accs , color='blue')
    plt.title('Accuracy vs depth.')
    plt.xlabel('depth')
    plt.ylabel('Accuracy')
    plt.show()
def width_of_network(depth=6):
    widths = [8,16,32,64]
    accs = []
    for width in widths:
        model = MLP(depth, width, input_dim, output_dim)
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, _ = train(train_dataset, val_dataset,
                                                                                                 test_dataset, model,
                                                                                                 lr=0.001, epochs=50,
                                                                                                 batch_size=256)
        accs.append(val_accs[-1])
    plt.figure()
    plt.plot(widths,accs , color='blue')
    plt.title('Accuracy vs Number of neurons')
    plt.xlabel('Number of neurons')
    plt.ylabel('Accuracy')
    plt.show()


def monitoring_gradients(train_dataset, val_dataset, test_dataset, lr=0.001, epochs=10, batch_size=256, width=4,
                         depth=100):
    layers = [0,30,60,90,95,99]
    model = MLP(depth, width, input_dim, output_dim)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    gradients = [[] for _ in range(epochs)]


    model.to(device)

    for ep in range(epochs):
        model.train()
        num_batches = 0
        gradients[ep] = [0]*depth

        for batch_idx, (samples, labels) in enumerate(trainloader):
            samples = samples.to(device)
            labels = labels.to(device)
            output = model(samples)
            loss = criterion(output, labels)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2**31)
            for layer in range(depth-1):
                gradients[ep][layer] += model.mid_layers[layer].weight.grad.norm(2).item() **2


            optimizer.step()

            # Store individual batch loss
            num_batches += 1

        gradients[ep] = [g/num_batches for g in gradients[ep]]
        print(f'Epoch {ep} gradients: {gradients[ep]}')
    avg_gradients = []
    for layer in layers:
        avg_gradients.append(np.mean([gradients[ep][layer] for ep in range(epochs)]))
    plt.plot(layers,avg_gradients)
    plt.title('Average gradients for different layers')
    plt.xlabel('Layer')
    plt.ylabel('Average gradient')
    plt.show()


def transform_to_20D(data):
    # Extract longitude and latitude columns
    features = data[['long', 'lat']].values

    # Create alphas from 0.1 to 1.0 in 10 steps
    alphas = np.linspace(0.1, 1.0, 10)

    # Initialize the transformed features array
    transformed_features = []

    # Apply sine transformations for each feature (long and lat)
    for feature_idx in range(features.shape[1]):
        feature_transforms = [np.sin(alpha * features[:, feature_idx]) for alpha in alphas]
        transformed_features.extend(feature_transforms)

    # Convert to numpy array and transpose to match original dataframe structure
    transformed_data = np.column_stack(transformed_features)

    # Create a new dataframe with transformed features
    transformed_df = pd.DataFrame(transformed_data,
                                  columns=[f'{col}_{alpha:.1f}' for col in ['long', 'lat'] for alpha in alphas])

    # Add back any other columns from the original dataframe
    for col in data.columns:
        if col not in ['long', 'lat']:
            transformed_df[col] = data[col]

    return transformed_df

def pre_processing():
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    train_data = EuropeDataset('train.csv')
    valid_data = EuropeDataset('validation.csv')
    test_data = EuropeDataset('test.csv')

    # Parallel sine transformations
    def apply_sine_transformations(samples):
        # List to store transformations for each alpha
        transformed_samples_list = []

        for alpha in alphas:
            # Apply sine transformation for each alpha
            transformed_samples = torch.sin(alpha * samples)
            transformed_samples_list.append(transformed_samples)

        # Concatenate transformations along the feature dimension
        return torch.cat(transformed_samples_list, dim=1)

    # Apply transformations
    transformed_train = apply_sine_transformations(train_data.samples)
    transformed_valid = apply_sine_transformations(valid_data.samples)
    transformed_test = apply_sine_transformations(test_data.samples)

    # Update dataset samples
    train_data.samples = transformed_train
    valid_data.samples = transformed_valid
    test_data.samples = transformed_test

    return train_data, valid_data, test_data

def compare_pre_processing(train_set,valid_set,test_set,lr=0.001,epochs=50,batch_size=256,width=16,depth=6):
    train_data,val_data,test_data = pre_processing()
    model = MLP(depth, width, 20, output_dim)
    default_run(model,train_dataset=train_data, val_dataset=val_data, test_dataset=test_data)
    model = MLP(depth, width, 2, output_dim)
    default_run(model,train_set,valid_set,test_set,lr,epochs,batch_size)





if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')
    output_dim = len(train_dataset.labels.unique())
    input_dim = train_dataset.samples.shape[1]




    #### YOUR CODE HERE #####
    # Find the number of classes, e.g.:

    # default_run()
    # different_lr(train_dataset, val_dataset, test_dataset, epochs=50, batch_size=256)
    # different_batch_size(train_dataset, val_dataset, test_dataset,lr=0.001)
    # different_depth_and_width(train_dataset, val_dataset, test_dataset,lr=0.001)
    # depth_of_network()
    # width_of_network()
    # monitoring_gradients(train_dataset, val_dataset, test_dataset, lr=0.001, epochs=10, batch_size=256,width=4,depth=100)
    compare_pre_processing(train_dataset, val_dataset, test_dataset)
