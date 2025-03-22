import models as md
import matplotlib.pyplot as plt
import helpers as hp
import numpy as np
import torch
from torch import nn
import copy
from sklearn.tree import DecisionTreeClassifier


def prepare_data(multi_class=False):
    if not multi_class: # Load the data
        X_train, _ = hp.read_data_demo('train.csv')
        X_valid, _ = hp.read_data_demo('validation.csv')
        X_test, _ = hp.read_data_demo('test.csv')
    else:
        X_train, _ = hp.read_data_demo('train_multiclass.csv')
        X_valid, _ = hp.read_data_demo('validation_multiclass.csv')
        X_test, _ = hp.read_data_demo('test_multiclass.csv')

    Y_train = X_train[:, -1]
    X_train = X_train[:, :-1]  # Remove the target column from features


    Y_valid = X_valid[:, -1]
    X_valid = X_valid[:, :-1]  # Remove the target column from features


    Y_test = X_test[:, -1]
    X_test = X_test[:, :-1]  # Remove the target column from features
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test
def accuracy(preds, Y):
    return np.mean(preds == Y)
def ridge_rigression():
    # Train and evaluate models
    lambdas = [0, 2, 4, 6, 8, 10]
    accuracy_valid = []
    accuracy_train = []
    accuracy_test = []

    for lambd in lambdas:
        ridge = md.Ridge_Regression(lambd)
        ridge.fit(X_train, Y_train)

        preds_valid = ridge.predict(X_valid)
        accuracy_valid.append(accuracy(preds_valid, Y_valid))

        preds_test = ridge.predict(X_test)
        accuracy_test.append(accuracy(preds_test, Y_test))

        preds_train = ridge.predict(X_train)
        accuracy_train.append(accuracy(preds_train, Y_train))

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, accuracy_valid, label='Validation', color='red')
    plt.plot(lambdas, accuracy_train, label='Train', color='blue')
    plt.plot(lambdas, accuracy_test, label='Test', color='green')
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    maX_model = md.Ridge_Regression(8)
    maX_model.fit(X_train, Y_train)
    print(maX_model.W)
    hp.plot_decision_boundaries(maX_model, X_test, Y_test, title='Max Model Decision Boundaries') # plot the best model
    min_model = md.Ridge_Regression(4)
    min_model.fit(X_train, Y_train)
    print(min_model.W)
    hp.plot_decision_boundaries(min_model, X_test, Y_test,
                                title='Min Model Decision Boundaries')  # plot the worst model
def target_function(x,y):
    return (x-3)**2 + (y-5)**2
def gradient_descent():
    W = [0,0]
    learning_rate = 0.1
    num_iterations = 1000
    optimized_vector_X = []
    optimized_vector_Y = []
    for i in range(num_iterations):
        optimized_vector_X.append(W[0])
        optimized_vector_Y.append(W[1])
        grad = np.array([2 * (W[0] - 3), 2 * (W[1] - 5)])
        W -= learning_rate * grad
        print("iteration ",i," W: ",W)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(optimized_vector_X, optimized_vector_Y,
                          c=range(len(optimized_vector_X)),
                          cmap='viridis',
                          s=50)
    plt.colorbar(scatter, label='Iteration')
    plt.plot(3, 5, 'r*', markersize=15, label='True minimum (3,5)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Descent Path')
    plt.legend()
    plt.grid(True)
    plt.show()


def logistic_regression(dataset, lr,num_epochs,batch_size,lambd=None,multiclass=False,ridge_regulation=False):
    decrease_lr = []
    best_validation_acc = 0
    best_model = None
    n_classes = len(np.unique(Y_train))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = md.Logistic_Regression(X_train.shape[1], n_classes).float()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    if multiclass:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    # Lists to store metrics during training
    history = {
        'train_losses': [],
        'train_accuracies': [],
        'validation_losses': [],
        'validation_accuracies': [],
        'test_losses': [],
        'test_accuracies': []
    }
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            print(outputs)
            print(labels)
            loss = criterion(outputs, labels)
            if ridge_regulation:
                l2_reg = 0
                for W in model.parameters():
                    l2_reg = l2_reg + W.norm(2)**2
                loss += lambd * l2_reg
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        if multiclass:
            decrease_lr.append(lr_scheduler.get_last_lr())
            lr_scheduler.step()



        # Calculate training metrics
        avg_train_loss = train_loss / len(dataloader)
        train_accuracy = 100 * correct_train / total_train
        history['train_losses'].append(avg_train_loss)
        history['train_accuracies'].append(train_accuracy)

        # Validation phase
        model.eval()
        with torch.no_grad():
            # Validation set
            valid_inputs = torch.tensor(X_valid).float().to(device)
            valid_labels = torch.tensor(Y_valid).long().to(device)
            valid_outputs = model(valid_inputs)
            valid_loss = criterion(valid_outputs, valid_labels)
            if ridge_regulation:
                l2_reg = 0
                for W in model.parameters():
                    l2_reg = l2_reg + W.norm(2) ** 2
                valid_loss += lambd * l2_reg
            _, predicted = torch.max(valid_outputs.data, 1)
            valid_accuracy = 100 * (predicted == valid_labels).sum().item() / len(Y_valid)

            history['validation_losses'].append(valid_loss.item())
            history['validation_accuracies'].append(valid_accuracy)

            # Test set
            test_inputs = torch.tensor(X_test).float().to(device)
            test_labels = torch.tensor(Y_test).long().to(device)
            test_outputs = model(test_inputs)
            test_loss = criterion(test_outputs, test_labels)
            if ridge_regulation:
                l2_reg = 0
                for W in model.parameters():
                    l2_reg = l2_reg + W.norm(2) ** 2
                test_loss += lambd * l2_reg
            _, predicted = torch.max(test_outputs.data, 1)
            test_accuracy = 100 * (predicted == test_labels).sum().item() / len(Y_test)

            history['test_losses'].append(test_loss.item())
            history['test_accuracies'].append(test_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {valid_loss.item():.4f}, Validation Accuracy: {valid_accuracy:.2f}%')
        print(f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy:.2f}%\n')



        # Save best model based on validation loss
        if valid_accuracy > best_validation_acc:
            best_validation_acc = valid_accuracy
            best_model = copy.deepcopy(model)
            torch.save({
                'model_state_dict': model.state_dict(),
                'history': history,
                'epoch': epoch,
                'lr': lr
            }, f'model_acc_{valid_accuracy:.2f}.pt')

    # Plot training curves

    plt.figure(figsize=(12, 4))

    # Plot losses
    plt.plot(history['train_losses'], label='Train')
    plt.plot(history['validation_losses'], label='Validation')
    plt.plot(history['test_losses'], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if not ridge_regulation:
        plt.title(f'Logistic Regression Learning Curve (lr={lr})')
    else:
        plt.title(f'Logistic Regression Learning Curve (lambda={lambd})')
    plt.legend()

    # Plot accuracies

    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracies'], label='Train')
    plt.plot(history['validation_accuracies'], label='Validation')
    plt.plot(history['test_accuracies'], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return best_validation_acc, best_model,history,decrease_lr



def logistic_regression_hyperparameter_tuning(learning_rates,X,Y,num_epochs,batch_size,multiclass=False,ridge_regulation=False):
    lambdas = [0,2,4,6,8,10]
    best_validation_acc = 0  # Changed from inf since we want highest accuracy
    best_overall_model = None
    best_history = None
    best_lr = None
    x_tensor = torch.tensor(X).float()
    y_tensor = torch.tensor(Y).long()
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    if not ridge_regulation:
        for lr in learning_rates:
            current_accuracy,current_model,history,decreasing_lr = logistic_regression(dataset,lr,num_epochs,batch_size,lambd=None,multiclass=True,ridge_regulation=ridge_regulation)
            if current_accuracy > best_validation_acc:
                best_validation_acc = current_accuracy
                best_overall_model = current_model
                best_history = history
                best_lr = decreasing_lr
    else:
        for lambd in lambdas:
            current_accuracy, current_model, history, decreasing_lr = logistic_regression(dataset, 0.01, num_epochs, batch_size,lambd,
                                                                                         multiclass=True, ridge_regulation=True)
            print("current accuracy: ",current_accuracy)
            print("best accuracy: ",best_validation_acc)
            if current_accuracy > best_validation_acc:
                best_validation_acc = current_accuracy
                best_overall_model = current_model
                best_history = history
                best_lr = decreasing_lr
    if best_history:
        # Plot the best model's metrics
        plt.figure(figsize=(12, 4))

        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(best_history['train_losses'], label='Train')
        plt.plot(best_history['validation_losses'], label='Validation')
        plt.plot(best_history['test_losses'], label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curves - Loss')
        plt.legend()

        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(best_history['train_accuracies'], label='Train')
        plt.plot(best_history['validation_accuracies'], label='Validation')
        plt.plot(best_history['test_accuracies'], label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Learning Curves - Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()
    return best_validation_acc,best_overall_model,best_history,best_lr

def decision_tree_classifier():
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = prepare_data(True)
    DecisionTreeClassifier_model_2 = DecisionTreeClassifier(max_depth=2)
    DecisionTreeClassifier_model_2.fit(X_train,Y_train)
    print("DecisionTreeClassifier_model accuracy on train data: ",DecisionTreeClassifier_model_2.score(X_train,Y_train))
    print("DecisionTreeClassifier_model accuracy on validation data: ",DecisionTreeClassifier_model_2.score(X_valid,Y_valid))
    print("DecisionTreeClassifier_model accuracy on test data: ",DecisionTreeClassifier_model_2.score(X_test,Y_test))
    hp.plot_decision_boundaries(DecisionTreeClassifier_model_2, X_test, Y_test, title='Decision Boundaries')
    DecisionTreeClassifier_model_10 = DecisionTreeClassifier(max_depth=10)
    DecisionTreeClassifier_model_10.fit(X_train,Y_train)
    print("DecisionTreeClassifier_model accuracy on train data: ",DecisionTreeClassifier_model_10.score(X_train,Y_train))
    print("DecisionTreeClassifier_model accuracy on validation data: ",DecisionTreeClassifier_model_10.score(X_valid,Y_valid))
    print("DecisionTreeClassifier_model accuracy on test data: ",DecisionTreeClassifier_model_10.score(X_test,Y_test))
    hp.plot_decision_boundaries(DecisionTreeClassifier_model_10, X_test, Y_test, title='Decision Boundaries')





# ridge_rigression()
# gradient_descent()

#2 classes

# X_train, Y_train, X_valid, Y_valid, X_test, Y_test = prepare_data(False)
# best_accuracy,best_model = logistic_regression_hyperparameter_tuning([0.1,0.01,0.001],X_train,Y_train,10,32)
# print(best_accuracy)
# hp.plot_decision_boundaries(best_model, X_test, Y_test, title='Best Model Decision Boundaries') # plot the best model

#multiclass

# X_train, Y_train, X_valid, Y_valid, X_test, Y_test = prepare_data(True)
# best_accuracy,best_model,best_history,best_lr = logistic_regression_hyperparameter_tuning([0.01,0.001,0.0003],X_train,Y_train,num_epochs=30,batch_size=32,multiclass=True)
# print(best_lr)
# plt.plot(best_lr,best_history['validation_accuracies'],label='validation accuracies',color='red')
# plt.plot(best_lr,best_history['test_accuracies'],label='test accuracies',color='blue')
# plt.xlabel('learning rate')
# plt.ylabel('accuracy')
# plt.legend()
# plt.show()



#desicio tree classifiers

# decision_tree_classifier()

#ridge regulation
X_train, Y_train, X_valid, Y_valid, X_test, Y_test = prepare_data(True)
logistic_regression(X_train,0.01,30,32,ridge_regulation=False)
# best_accuracy,best_model,best_history,best_lr = logistic_regression_hyperparameter_tuning([0.01],X_train,Y_train,num_epochs=30,batch_size=32,multiclass=True,ridge_regulation=True)
# print("best model out",best_model)
# hp.plot_decision_boundaries(best_model, X_test, Y_test, title='Best Model Decision Boundaries')




