import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
from torchvision import transforms
from PIL import Image


def plot_decision_boundaries(model, X, y, title='Decision Boundaries', implicit_repr=False):
    """
    Plots decision boundaries of a classifier and colors the space by the prediction of each point.

    Parameters:
    - model: The trained classifier (sklearn model).
    - X: Numpy Feature matrix.
    - y: Numpy array of Labels.
    - title: Title for the plot.
    """
    # h = .02  # Step size in the mesh

    # enumerate y
    y_map = {v: i for i, v in enumerate(np.unique(y))}
    enum_y = np.array([y_map[v] for v in y]).astype(int)

    h_x = (np.max(X[:, 0]) - np.min(X[:, 0])) / 200
    h_y = (np.max(X[:, 1]) - np.min(X[:, 1])) / 200

    # Plot the decision boundary.
    added_margin_x = h_x * 20
    added_margin_y = h_y * 20
    x_min, x_max = X[:, 0].min() - added_margin_x, X[:, 0].max() + added_margin_x
    y_min, y_max = X[:, 1].min() - added_margin_y, X[:, 1].max() + added_margin_y
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))

    # Make predictions on the meshgrid points.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if implicit_repr:
        model_inp = np.c_[xx.ravel(), yy.ravel()]
        new_model_inp = np.zeros((model_inp.shape[0], model_inp.shape[1] * 10))
        alphas = np.arange(0.1, 1.05, 0.1)
        for i in range(model_inp.shape[1]):
            for j, a in enumerate(alphas):
                new_model_inp[:, i * len(alphas) + j] = np.sin(a * model_inp[:, i])
        model_inp = torch.tensor(new_model_inp, dtype=torch.float32, device=device)
    else:
        model_inp = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=device)
    with torch.no_grad():
        Z = model(model_inp).argmax(dim=1).cpu().numpy()
    Z = np.array([y_map[v] for v in Z])
    Z = Z.reshape(xx.shape)
    vmin = np.min([np.min(enum_y), np.min(Z)])
    vmax = np.min([np.max(enum_y), np.max(Z)])

    # Plot the decision boundary.
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8, vmin=vmin, vmax=vmax)

    # Scatter plot of the data points with matching colors.
    plt.scatter(X[:, 0], X[:, 1], c=enum_y, cmap=plt.cm.Paired, edgecolors='k', s=40, alpha=0.7, vmin=vmin, vmax=vmax)

    plt.title("Decision Boundaries")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.show()


def read_data_demo(filename='train.csv'):
    """
    Read the data from the csv file and return the features and labels as numpy arrays.
    """

    # the data in pandas dataframe format
    df = pd.read_csv(filename)

    # extract the column names
    col_names = list(df.columns)

    # the data in numpy array format
    data_numpy = df.values

    return data_numpy, col_names





def load_images(directory, num_samples_per_class=100, train=True):
    """
    Load images from a specified directory with a structured format.

    Parameters:
    -----------
    directory : str
        Root directory containing train/validation/test subdirectories
    num_samples_per_class : int, optional
        Number of samples to load from each class (default 100 for train, 20 for val/test)
    train : bool, optional
        Whether to load training data (True) or validation/test data (False)

    Returns:
    --------
    tuple: (images, labels)
        images: numpy array of image data
        labels: numpy array of corresponding labels
    """
    # Set sample size based on train/validation/test
    samples_per_class = num_samples_per_class if train else 20

    # Modes to check
    modes = ['train', 'val', 'test']

    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to a consistent size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    # Lists to store images and labels
    images = []
    labels = []

    # Loop through the specified mode
    current_mode = 'train' if train else 'val' if 'val' in modes else 'test'
    mode_path = os.path.join(directory, current_mode)

    # Classes to load
    classes = ['0_real', '1_fake']

    # Load images for each class
    for label, class_name in enumerate(classes):
        class_path = os.path.join(mode_path, class_name)

        # Get list of image files
        image_files = [f for f in os.listdir(class_path) if
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        # Randomly select images
        selected_files = np.random.choice(image_files, min(samples_per_class, len(image_files)), replace=False)

        # Load and process images
        for img_file in selected_files:
            img_path = os.path.join(class_path, img_file)
            try:
                # Open image
                img = Image.open(img_path).convert('RGB')

                # Apply transformations
                img_tensor = transform(img)

                # Convert to numpy and append
                images.append(img_tensor.numpy())
                labels.append(label)

            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels


# Usage example
def load_dataset(directory):
    """
    Load train, validation, and test datasets

    Parameters:
    -----------
    directory : str
        Root directory containing image datasets

    Returns:
    --------
    tuple: (train_images, train_labels, val_images, val_labels, test_images, test_labels)
    """
    # Load training data
    train_images, train_labels = load_images(directory, train=True)

    # Load validation data
    val_images, val_labels = load_images(directory, train=False)

    # Load test data
    test_images, test_labels = load_images(directory, train=False)

    return train_images, train_labels, val_images, val_labels, test_images, test_labels

# Example of how to use the function
