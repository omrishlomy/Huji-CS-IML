import torch
import torch.nn as nn
from dataset import EuropeDataset
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import numpy as np
def normalize_tensor(tensor, d):
    """
    Normalize the input tensor along the specified axis to have a mean of 0 and a std of 1.
    
    Parameters:
        tensor (torch.Tensor): Input tensor to normalize.
        d (int): Axis along which to normalize.
    
    Returns:
        torch.Tensor: Normalized tensor.
    """
    mean = torch.mean(tensor, dim=d, keepdim=True)
    std = torch.std(tensor, dim=d, keepdim=True)
    normalized = (tensor - mean) / std
    return normalized


class GMM(nn.Module):
    def __init__(self, n_components):
        """
        Gaussian Mixture Model in 2D using PyTorch.

        Args:
            n_components (int): Number of Gaussian components.
        """
        super().__init__()        
        self.n_components = n_components

        # Mixture weights (logits to be softmaxed)
        self.weights = nn.Parameter(torch.randn(n_components))

        # Means of the Gaussian components (n_components x 2 for 2D data)
        self.means = nn.Parameter(torch.randn(n_components, 2))

        # Log of the variance of the Gaussian components (n_components x 2 for 2D data)
        self.log_variances = nn.Parameter(torch.zeros(n_components, 2))  # Log-variances (diagonal covariance)

    def compute_conditional(self, X, i):
        """
        Compute the log probability of X given the i-th Gaussian component.
        Args:
            X (torch.Tensor): Input data of shape (n_samples, 2).
            i (int): Index of the Gaussian component.

        Returns:
            torch.Tensor: Log conditional probability of shape (n_samples,).
        """
        # Use log variances directly
        log_var_x = self.log_variances[i][0]
        log_var_y = self.log_variances[i][1]

        # Log probability components
        log_const = -math.log(2 * math.pi)
        log_var_terms = -log_var_x - log_var_y

        # Squared terms with variance normalization
        diff_x = X[:, 0] - self.means[i][0]
        diff_y = X[:, 1] - self.means[i][1]
        squared_terms = -0.5 * ((diff_x ** 2 / torch.exp(log_var_x)) +
                                (diff_y ** 2 / torch.exp(log_var_y)))

        # Combine all terms
        log_prob = log_const + log_var_terms + squared_terms

        return log_prob

    def forward(self, X):
        """
        Compute the log-likelihood of the data.
        Args:
            X (torch.Tensor): Input data of shape (n_samples, 2).

        Returns:
            torch.Tensor: Log-likelihood of shape (n_samples,).
        """
        log_pk = nn.functional.log_softmax(self.weights, dim=0)
        logs = []
        for i in range(self.n_components):
            log_pxk = self.compute_conditional(X, i)
            logs.append(log_pxk + log_pk[i])
        logs = torch.stack(logs, dim=1)
        return torch.logsumexp(logs, dim=1)


    def loss_function(self, log_likelihood):
        """
        Compute the negative log-likelihood loss.
        Args:
            log_likelihood (torch.Tensor): Log-likelihood of shape (n_samples,).

        Returns:
            torch.Tensor: Negative log-likelihood.
        """
        return -torch.mean(log_likelihood)


    def sample(self, n_samples):
        """
        Generate samples from the GMM model.
        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        gaussian = torch.multinomial(nn.functional.softmax(self.weights, dim=0), n_samples, replacement=True)
        samples = torch.randn(n_samples,2)
        variance = torch.exp(self.log_variances)
        for i in range(n_samples):
            idx = gaussian[i]
            # Convert log variance to standard deviation: sqrt(exp(log_var))
            std_dev = torch.sqrt(torch.exp(self.log_variances[idx]))
            samples[i] = samples[i] * std_dev + self.means[idx]

        return samples
    
    def conditional_sample(self, n_samples, label):
        """
        Generate samples from a specific uniform component.
        Args:
            n_samples (int): Number of samples to generate.
            label (int): Component index.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        samples = torch.randn(n_samples, 2)
        # Convert log variances to standard deviations
        std_dev = torch.sqrt(torch.exp(self.log_variances[label]))
        # Scale by std dev and shift by mean
        samples = samples * std_dev + self.means[label]
        return samples


class UMM(nn.Module):
    def __init__(self, n_components):
        """
        Uniform Mixture Model in 2D using PyTorch.

        Args:
            n_components (int): Number of uniform components.
        """
        super().__init__()        
        self.n_components = n_components

        # Mixture weights (logits to be softmaxed)
        self.weights = nn.Parameter(torch.randn(n_components))

        # Center value of the uniform components (n_components x 2 for 2D data)
        self.centers = nn.Parameter(torch.randn(n_components, 2))

        # Log of size of the uniform components (n_components x 2 for 2D data)
        self.log_sizes = nn.Parameter(torch.log(torch.ones(n_components, 2) + torch.rand(n_components, 2)*0.2))

    def forward(self, X):
        log_pk = nn.functional.log_softmax(self.weights, dim=0)
        sizes = torch.exp(self.log_sizes)
        logs = []

        for k in range(self.n_components):
            # Bounds check
            in_bounds_x = torch.logical_and(
                X[:, 0] >= (self.centers[k][0] - sizes[k][0] / 2),
                X[:, 0] <= (self.centers[k][0] + sizes[k][0] / 2)
            )
            in_bounds_y = torch.logical_and(
                X[:, 1] >= (self.centers[k][1] - sizes[k][1] / 2),
                X[:, 1] <= (self.centers[k][1] + sizes[k][1] / 2)
            )
            in_bounds = torch.logical_and(in_bounds_x, in_bounds_y)

            # Set probabilities
            log_pxk = torch.full_like(X[:, 0], 1*math.e**-6, dtype=torch.float)

            # Only compute inside probability if there are points inside
            if torch.any(in_bounds):
                log_uniform_prob = -torch.log(sizes[k][0]) - torch.log(sizes[k][1])
                log_pxk[in_bounds] = log_uniform_prob

            logs.append(log_pxk + log_pk[k])

        logs = torch.stack(logs, dim=1)
        return torch.logsumexp(logs, dim=1)
        
    
    
    def loss_function(self, log_likelihood):
        """
        Compute the negative log-likelihood loss.
        Args:
            log_likelihood (torch.Tensor): Log-likelihood of shape (n_samples,).

        Returns:
            torch.Tensor: Negative log-likelihood.
        """
        return -torch.mean(log_likelihood)


    def sample(self, n_samples):
        """
        Generate samples from the UMM model.
        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        gaussian = torch.multinomial(nn.functional.softmax(self.weights, dim=0), n_samples, replacement=True)
        sizes = torch.exp(self.log_sizes)
        samples = []
        for i in range(n_samples):
            idx = gaussian[i]
            uniform = (torch.distributions.uniform.Uniform(low=self.centers[idx] - sizes[idx]/2, high=self.centers[idx] + sizes[idx]/2).sample())
            samples.append(uniform)
        return torch.stack(samples)

    def conditional_sample(self, n_samples, label):
        """
        Generate samples from a specific uniform component.
        Args:
            n_samples (int): Number of samples to generate.
            label (int): Component index.

        Returns:
            torch.Tensor: Generated samples of shape (n_samples, 2).
        """
        sizes = torch.exp(self.log_sizes)
        samples = torch.distributions.uniform.Uniform(low=self.centers[label] - sizes[label]/2, high=self.centers[label] + sizes[label]/2).sample((n_samples,))
        return samples


def train(model, train_loader, test_loader, num_epochs, learning_rate,is_mean=False,is_Umm=False):
    """
    Train the model using the given data loaders and hyperparameters.

    Parameters:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Testing data loader.
        num_epochs (int): Number of epochs.
        learning_rate (float): Learning rate.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for i, data in enumerate(train_loader):
            features, _ = data  # We don't need labels for GMM
            optimizer.zero_grad()

            # Forward pass - get log likelihoods
            log_likelihoods = model(features)

            # Compute loss
            loss = model.loss_function(log_likelihoods)

            # Backward pass
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Testing phase
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                features, _ = data
                log_likelihoods = model(features)
                loss = model.loss_function(log_likelihoods)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
    if is_mean:
        if not is_Umm:
            model_path = os.path.join(models_dir, f'gmm_{model.n_components}_{num_epochs}_epochs_components_Mean.pt')
        else:
            model_path = os.path.join(models_dir, f'umm_{model.n_components}_{num_epochs}_epochs_components_Mean.pt')
    else:
        if not is_Umm:
            model_path = os.path.join(models_dir, f'gmm_{model.n_components}_{num_epochs}_epochs_components.pt')
        else:
            model_path = os.path.join(models_dir, f'umm_{model.n_components}_{num_epochs}_epochs_components.pt')


    torch.save({
        'model_state_dict': model.state_dict(),
        'n_components': model.n_components
    }, model_path)
    return model
def plot_differnet_gaussians(model, n_samples,is_Umm=False):
    samples = model.sample(n_samples)
    samples_np = samples.detach().numpy()
    plt.scatter(samples_np[:, 0], samples_np[:, 1])
    if not is_Umm:
        plt.title(f'GMM with {model.n_components} components')
    else:
        plt.title(f'UMM with {model.n_components} components')
    plt.xlabel('X')
    plt.ylabel('Y')
def plot_single_gaussian(model, n_samples,n_components,is_Umm=False):
    colors = cm.tab10(np.linspace(0, 1, n_components))
    for component in range(n_components):
        samples = model.conditional_sample(n_samples, component)
        samples_np = samples.detach().numpy()
        if not is_Umm:
            plt.title(f'GMM with {model.n_components} components')
        else:
            plt.title(f'UMM with {model.n_components} components')
        plt.scatter(samples_np[:, 0], samples_np[:, 1], label=f'Component {component}', color=colors[component])
        plt.xlabel('X')
        plt.ylabel('Y')
    plt.show()
def train_components(num_epochs=50,is_Umm=False):
    lr = 0.01 if not is_Umm else 0.001
    components = [1,5,10,33]
    for component in range(len(components)):
        if not is_Umm:
            model = GMM(n_components=components[component])
            model_path = os.path.join(models_dir, f'gmm_{components[component]}_components.pt')
        else:
            model = UMM(n_components=components[component])
            model_path = os.path.join(models_dir, f'umm_{components[component]}_epochs_components.pt')
        model = train(model, train_loader, test_loader, num_epochs, learning_rate=lr,is_mean=False,is_Umm=is_Umm)
    return trained_models

def plot_for_components(is_mean=False,is_Umm=False):
    components = [1, 5, 10, 33]
    colors = cm.tab10(np.linspace(0, 1, n_classes+1))

    # Create two separate figures
    plt.figure(figsize=(15, 5))  # First figure for regular samples

    # Regular samples
    for i, component in enumerate(components):
        # Calculate position in 1x6 grid
        plt.subplot(1, 6, i + 1)

        # Load model
        if not is_mean:
            if not is_Umm:
                checkpoint = torch.load(f'trained_models/gmm_{component}_{num_epochs}_epochs_components.pt')
                n_classes_model = GMM(n_components=checkpoint['n_components'])
                n_classes_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                checkpoint = torch.load(f'trained_models/umm_{component}_{num_epochs}_epochs_components.pt')
                n_classes_model = UMM(n_components=checkpoint['n_components'])
                n_classes_model.load_state_dict(checkpoint['model_state_dict'])

        else:
            if not is_Umm:
                checkpoint = torch.load(f'trained_models/gmm_{component}_{num_epochs}_epochs_components_Mean.pt')
                n_classes_model = GMM(n_components=checkpoint['n_components'])
                n_classes_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                checkpoint = torch.load(f'trained_models/umm_{component}_{num_epochs}_epochs_components_Mean.pt')
                n_classes_model = UMM(n_components=checkpoint['n_components'])
                n_classes_model.load_state_dict(checkpoint['model_state_dict'])

        # n_classes_model.eval()

        # Plot regular samples
        samples = n_classes_model.sample(1000)
        samples_np = samples.detach().numpy()
        plt.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5)
        plt.title(f'Regular samples\n{component} components')
        plt.xlabel('X')
        plt.ylabel('Y')
    plt.tight_layout()
    plt.show()

    # Create new figure for conditional samples
    plt.figure(figsize=(15, 5))  # Second figure

    # Conditional samples
    for i, component in enumerate(components):
        plt.subplot(1, 6, i + 1)
        for k in range(component):
            samples = n_classes_model.conditional_sample(100,k)
            samples_np = samples.detach().numpy()
            plt.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5,color=colors[k], s=20)
        plt.title(f'Regular samples\n{component} components')
        plt.xlabel('X')
        plt.ylabel('Y')
    plt.tight_layout()
    plt.show()



def plot_for_epochs(is_mean=False,is_Umm=False):
    epochs = [1, 10, 20, 30, 40, 50]
    colors = cm.tab10(np.linspace(0, 1, n_classes))

    # Create two separate figures
    plt.figure(figsize=(15, 5))  # First figure for regular samples

    # Regular samples
    for i, epoch in enumerate(epochs):
        # Calculate position in 1x6 grid
        plt.subplot(1, 6, i + 1)

        # Load model
        if not is_mean:
            if not is_Umm:
                checkpoint = torch.load(f'trained_models/gmm_33_{epoch}_epochs_components.pt')
                n_classes_model = GMM(n_components=checkpoint['n_components'])
                n_classes_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                checkpoint = torch.load(f'trained_models/umm_33_{epoch}_epochs_components.pt')
                n_classes_model = UMM(n_components=checkpoint['n_components'])
                n_classes_model.load_state_dict(checkpoint['model_state_dict'])

        else:
            if not is_Umm:
                checkpoint = torch.load(f'trained_models/gmm_33_{epoch}_epochs_components_Mean.pt')
                n_classes_model = GMM(n_components=checkpoint['n_components'])
                n_classes_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                checkpoint = torch.load(f'trained_models/umm_33_{epoch}_epochs_components_Mean.pt')
                n_classes_model = UMM(n_components=checkpoint['n_components'])
                n_classes_model.load_state_dict(checkpoint['model_state_dict'])

        # n_classes_model.eval()

        # Plot regular samples
        samples = n_classes_model.sample(1000)
        samples_np = samples.detach().numpy()
        plt.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5)
        plt.title(f'Regular samples\n{epoch} epochs')
        plt.xlabel('X')
        plt.ylabel('Y')
    plt.tight_layout()

    # Create new figure for conditional samples
    plt.figure(figsize=(15, 5))  # Second figure

    # Conditional samples
    for i, epoch in enumerate(epochs):
        plt.subplot(1, 6, i + 1)

        # Load model (reusing the model from above)
        if not is_mean:
            if not is_Umm:
                checkpoint = torch.load(f'trained_models/gmm_33_{epoch}_epochs_components.pt')
                n_classes_model = GMM(n_components=checkpoint['n_components'])
                n_classes_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                checkpoint = torch.load(f'trained_models/umm_33_{epoch}_epochs_components.pt')
                n_classes_model = UMM(n_components=checkpoint['n_components'])
                n_classes_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            if not is_Umm:
                checkpoint = torch.load(f'trained_models/gmm_33_{epoch}_epochs_components_Mean.pt')
                n_classes_model = GMM(n_components=checkpoint['n_components'])
                n_classes_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                checkpoint = torch.load(f'trained_models/umm_33_{epoch}_epochs_components_Mean.pt')
                n_classes_model = UMM(n_components=checkpoint['n_components'])
                n_classes_model.load_state_dict(checkpoint['model_state_dict'])

        n_classes_model.eval()

        for component in range(n_classes):
            samples = n_classes_model.conditional_sample(100, component)
            samples_np = samples.detach().numpy()
            plt.scatter(samples_np[:, 0], samples_np[:, 1],
                        color=colors[component], alpha=0.5, s=20)
        plt.title(f'Conditional samples\n{epoch} epochs')
        plt.xlabel('X')
        plt.ylabel('Y')
    plt.tight_layout()

    plt.show()  # This will show both figures


def calculate_log_likelihood(model, data_loader):
    model.eval()
    total_log_likelihood = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (features, _) in enumerate(data_loader):
            # Get log likelihoods for this batch
            log_likelihoods = model(features)
            total_log_likelihood += log_likelihoods.sum() / batch_size
            num_batches += 1

    return total_log_likelihood / num_batches
def calculate_log_liklyhood_for_epochs(is_mean=False,is_Umm=False):
    train_liklihoods = []
    test_liklihoods = []
    for epoch in [1,10,20,30,40,50]:
        if not is_Umm:
            model = GMM(n_classes)
        else:
            model = UMM(n_classes)
        if not is_mean:
            if not is_Umm:
                checkpoint = torch.load(f'trained_models/gmm_{n_classes}_{epoch}_epochs_components.pt')
            else:
                checkpoint = torch.load(f'trained_models/umm_{n_classes}_{epoch}_epochs_components.pt')
        else:
            if not is_Umm:
                checkpoint = torch.load(f'trained_models/gmm_{n_classes}_{epoch}_epochs_components_Mean.pt')
            else:
                checkpoint = torch.load(f'trained_models/umm_{n_classes}_{epoch}_epochs_components_Mean.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        train_liklihood = calculate_log_likelihood(model, train_loader)
        model.eval()
        test_liklihood = calculate_log_likelihood(model, test_loader)
        train_liklihoods.append(train_liklihood)
        test_liklihoods.append(test_liklihood)
    plt.plot([1,10,20,30,40,50], train_liklihoods, label='Train',color='red')
    plt.plot([1,10,20,30,40,50], test_liklihoods, label='Test',color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Log Likelihood')
    plt.title('Log Likelihood vs. Epochs')
    plt.legend()
    plt.show()
def get_mean_locations(train_data):
    means = []
    for i in range(n_classes):
        # Get all samples for class i and compute their mean
        class_samples = train_data.features[train_data.labels == i]
        mean = class_samples.mean(dim=0)
        means.append(mean)
    return nn.Parameter(torch.stack(means))
def train_GMM():
    # for bool in [True,False]:
    #     for epoch in [1,10,20,30,40,50]:
    #         model = GMM(n_classes)
    #         if bool:
    #             with torch.no_grad():
    #                 model.means.copy_(normalize_tensor(get_mean_locations(train_dataset),0))
    #         model = train(model, train_loader, test_loader, epoch, learning_rate=0.01,is_mean=bool,is_Umm=False)
    plot_for_epochs(is_mean=True,is_Umm=False)
    calculate_log_liklyhood_for_epochs(is_mean=True,is_Umm=False)
    plot_for_epochs(is_mean=False,is_Umm=False)
    calculate_log_liklyhood_for_epochs(is_mean=False,is_Umm=False)
def train_UMM():
    # for bool in [True,False]:
    #     for epoch in [1,10,20,30,40,50]:
    #         model = UMM(n_classes)
    #         if bool:
    #             with torch.no_grad():
    #                 model.centers.copy_(normalize_tensor(get_mean_locations(train_dataset),0))
    #         model = train(model, train_loader, test_loader, epoch, learning_rate=0.001,is_mean=bool,is_Umm=True)
    plot_for_epochs(is_mean=True,is_Umm=True)
    calculate_log_liklyhood_for_epochs(is_mean=True,is_Umm=True)
    plot_for_epochs(is_mean=False,is_Umm=True)
    calculate_log_liklyhood_for_epochs(is_mean=False,is_Umm=True)

if __name__ == "__main__":
    
    torch.manual_seed(42)
    train_dataset = EuropeDataset('train.csv')
    test_dataset = EuropeDataset('test.csv')

    batch_size = 4096
    num_epochs = 50
    # Use Adam optimizer
    #TODO: Determine learning rate
    # learning_rate for GMM = 0.01
    # learning_rate for UMM = 0.001
    
    train_dataset.features = normalize_tensor(train_dataset.features, d=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataset.features = normalize_tensor(test_dataset.features, d=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    n_classes = train_dataset.labels.unique().shape[0]
    components = [1,5,10,n_classes]
    is_Umm = False #change to False for GMM
    models_dir = 'trained_models'  # Directory to save models
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if is_Umm:
        train_UMM()
    else:
        train_GMM()
    # train_components(50,True)
    # plot_for_components(False,True)



    



