import os
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import sklearn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += shortcut
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, pretrained=False, probing=False):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
            self.resnet18 = resnet18(weights=weights)
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            self.resnet18 = resnet18()

        self.first_layer = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.mid_layer_1 = self.make_layer(BasicBlock, 64, 2, 1)
        self.mid_layer_2 = self.make_layer(BasicBlock, 128, 2, 2)
        self.mid_layer_3 = self.make_layer(BasicBlock, 256, 2, 2)
        self.mid_layer_4 = self.make_layer(BasicBlock, 512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        in_features_dim = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()
        if probing:
            for name, param in self.resnet18.named_parameters():
                param.requires_grad = False
        self.logistic_regression = nn.Linear(in_features_dim, 1)

    def make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        # First block may change dimensions
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels  # Add this line! Update in_channels

        # Subsequent blocks
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))  # Use out_channels as in_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.resnet18(x)
        x = self.first_layer(x)
        x = self.mid_layer_1(x)
        x = self.mid_layer_2(x)
        x = self.mid_layer_3(x)
        x = self.mid_layer_4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.logistic_regression(x)
        return x


def get_loaders(path, transform, batch_size):
    """
    Get the data loaders for the train, validation and test sets.
    :param path: The path to the 'whichfaceisreal' directory.
    :param transform: The transform to apply to the images.
    :param batch_size: The batch size.
    :return: The train, validation and test data loaders.
    """
    train_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'train'), transform=transform)
    val_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'val'), transform=transform)
    test_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def compute_accuracy(model, data_loader, device):
    """
    Compute the accuracy of the model on the data in data_loader
    :param model: The model to evaluate.
    :param data_loader: The data loader.
    :param device: The device to run the evaluation on.
    :return: The accuracy of the model on the data in data_loader
    """
    model.eval()
    total_acc = 0
    with torch.no_grad():
        acc = 0
        num_of_batches = 0
        for samples, labels in data_loader:
            samples = samples.to(device)
            labels = labels.float().to(device)
            output = model(samples)
            output = output.squeeze()
            loss = criterion(output, labels)
            predictions = (output > 0.5).float()
            acc = (predictions == labels).float().mean().item()
            total_acc += acc
            num_of_batches += 1

    total_acc = total_acc / num_of_batches
    return total_acc


def run_training_epoch(model, criterion, optimizer, train_loader, device):
    """
    Run a single training epoch
    :param model: The model to train
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param train_loader: The data loader
    :param device: The device to run the training on
    :return: The average loss for the epoch.
    """
    model.train()
    train_loss = 0
    num_of_batches = 0
    for (imgs, labels) in tqdm(train_loader, total=len(train_loader)):
        imgs = imgs.to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        output = model(imgs)
        output = output.squeeze()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        num_of_batches += 1
    return train_loss / num_of_batches
def train_worst_model():
    worst_model = ResNet18(pretrained=False, probing=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    worst_model = worst_model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(worst_model.parameters(), lr=0.01)
    worst_model.train()
    for epoch in range(5):
        for (imgs, labels) in tqdm(train_loader, total=len(train_loader)):
            imgs = imgs.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            output = worst_model(imgs)
            output = output.squeeze()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    return worst_model
def train_best_models():
    best_model = ResNet18(pretrained=False, probing=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model = bess_model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(bess_model.parameters(), lr=0.0001)
    best_model.train()
    for epoch in range(5):
        for (imgs, labels) in tqdm(train_loader, total=len(train_loader)):
            imgs = imgs.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            output = best_model(imgs)
            output = output.squeeze()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    return best_model
def show_5_img(test_loader):
    best_model = train_best_models()
    worst_model = train_worst_model()
    best_model.eval()
    worst_model.eval()
    count_img = 0
    plt.figure(figsize=(15, 10))

    with torch.no_grad():
        for imgs, labels in test_loader:
            if count_img >= 5:
                break

            imgs = imgs.to(device)
            labels = labels.float().to(device)

            # Get predictions from both models
            output_best = best_model(imgs)
            output_worst = worst_model(imgs)
            output_best = output_best.squeeze()
            output_worst = output_worst.squeeze()

            predictions_best = (output_best > 0.5).float()
            predictions_worst = (output_worst > 0.5).float()

            # Check all images in the batch
            for i in range(len(labels)):
                if count_img >= 5:
                    break

                if predictions_best[i] == labels[i] and predictions_worst[i] != labels[i]:
                    plt.subplot(1, 5, count_img + 1)
                    plt.imshow(imgs[i].permute(1, 2, 0).cpu().numpy())
                    plt.title(
                        f'True: {labels[i].item():.0f}\nBest: {predictions_best[i].item():.0f}\nWorst: {predictions_worst[i].item():.0f}')
                    plt.axis('off')
                    count_img += 1

    plt.tight_layout()
    plt.show()

    if count_img == 0:
        print("No images found where best model was correct and worst model was wrong")



# Set the random seed for reproducibility
torch.manual_seed(0)

### UNCOMMENT THE FOLLOWING LINES TO TRAIN THE MODEL ###
# From Scratch
# model = ResNet18(pretrained=False, probing=False)
# Linear probing
model = ResNet18(pretrained=True, probing=True)
# Fine-tuning
# model = ResNet18(pretrained=True, probing=False)

transform = model.transform
batch_size = 32
num_of_epochs = 50
learning_rate = 0.0001
path = r"C:\Users\user\Desktop\Studies\מדעי המחשב\IML\exercises\ex4\whichfaceisreal"
train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)

# Train the model
accuracies = {"scracth": [], "probing": [], "fine_tuning": []}
models = {"scracth": ResNet18(pretrained=False, probing=False), "probing": ResNet18(pretrained=True, probing=True),
          "fine_tuning": ResNet18(pretrained=True, probing=False)}
best_model = None
worst_model = None
best_accuracy = 0
worst_accuracy = 1
for model_type in models:
    model = models[model_type]
    for lr in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        print("starting training f" + model_type + " with lr " + str(lr) + "...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        ### Define the loss function and the optimizer
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_acc = 0
        epochs_without_improvment = 0
        for epoch in range(num_of_epochs):
            # Run a training epoch
            loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
            # Compute the accuracy
            train_acc = compute_accuracy(model, train_loader, device)
            # Compute the validation accuracy
            val_acc = compute_accuracy(model, val_loader, device)
            if val_acc > best_acc:
                best_acc = val_acc
                epochs_without_improvment = 0
            else:
                epochs_without_improvment += 1
            if epochs_without_improvment > 5:
                print(f'Early stopping triggered after epoch {epoch + 1}')
                break

            print(f'Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss:.4f}, Val accuracy: {val_acc:.4f}')
        test_acc = compute_accuracy(model, test_loader, device)  # Compute the test accuracy
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = model
            torch.save(model.state_dict(), f"{model_type}_best_model.pth")
        if test_acc < worst_accuracy:
            worst_accuracy = test_acc
            worst_model = model
            torch.save(model.state_dict(), f"{model_type}_worst_model.pth")
        accuracies[model_type].append(test_acc)
print(accuracies)
torch.save(best_model.state_dict(), "best_model.pth")
torch.save(worst_model.state_dict(), "worst_model.pth")





