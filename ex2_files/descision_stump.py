import numpy as np
import matplotlib.pyplot as plt

def samples_to_update(samples,threshold,s):
        if s == "left":
            return samples < threshold
        else:
            return samples >= threshold
def compute_accuracy(y_true,y_pred):
    return np.mean(y_true == y_pred)


def decision_stump(X, y, scores):
    best_accuracy = 0
    best_scores = scores.copy()
    best_feature = None
    best_threshold = None
    best_class = None
    best_side = None

    for d in range(X.shape[1]):
        thresholds = np.random.uniform(np.min(X[:, d]), np.max(X[:, d]), 25)
        for theta in thresholds:
            for c in range(scores.shape[1]):
                for s in {"left", "right"}:
                    current_score_diff = np.zeros(shape=(scores.shape[0], scores.shape[1]))
                    update_samples = samples_to_update(X[:, d], theta, s)

                    if s == "right":
                        current_score_diff[update_samples] = np.eye(scores.shape[1])[c]
                    else:
                        current_score_diff[update_samples] = np.ones(scores.shape[1]) / scores.shape[1]

                    updated_scores = scores + current_score_diff
                    predictions = np.argmax(updated_scores, axis=1)
                    current_accuracy = compute_accuracy(y, predictions)

                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        best_scores = updated_scores.copy()
                        best_feature = d
                        best_threshold = theta
                        best_class = c
                        best_side = s

    return best_scores, best_accuracy, {
        'feature': best_feature,
        'threshold': best_threshold,
        'class': best_class,
        'side': best_side
    }
def decision_classifier(X,y,stumps):
    scores = np.zeros(shape=(X.shape[0], len(np.unique(y))))
    all_parameters = []
    for i in range(stumps):
        scores,accuracy,parameters = decision_stump(X,y,scores)
        all_parameters.append(parameters)
    return scores,accuracy,all_parameters


def predict(X_test, all_parameters, n_classes):
    scores = np.zeros(shape=(X_test.shape[0], n_classes))

    for parameters in all_parameters:  # Loop through all parameters
        current_score_diff = np.zeros_like(scores)

        feature = parameters['feature']
        threshold = parameters['threshold']
        side = parameters['side']

        update_samples = samples_to_update(X_test[:, feature], threshold, side)

        if side == "right":
            current_score_diff[update_samples] = np.eye(n_classes)[parameters['class']]
        else:
            current_score_diff[update_samples] = np.ones(n_classes) / n_classes

        scores += current_score_diff

    return np.argmax(scores, axis=1)

def plot_graph(X,y,X_test,y_test):
    max_stump = 30
    accuracies = []
    accuracies_test = []
    scores = np.zeros(shape=(X.shape[0], len(np.unique(y))))
    for stump in range(max_stump):
        scores,accuracy,parameters = decision_stump(X,y,scores)
        accuracies.append(accuracy)
        prediction_test = predict(X_test,parameters,len(np.unique(y)))
        accuracy_test = compute_accuracy(y_test,prediction_test)
        accuracies_test.append(accuracy_test)
    plt.plot(np.arange(max_stump),accuracies,label="train",color="red")
    plt.plot(np.arange(max_stump),accuracies_test,label="test",color="blue")
    plt.xlabel("Number of stumps")
    plt.ylabel("Accuracy")
    plt.title("Number of stumps vs Accuracy")
    plt.legend()
    plt.show()


def scatter_plot(X, y):
    stumps = [1,5,15,25]
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()

    for idx, stump in enumerate(stumps):
        scores, accuracy, all_parameters = decision_classifier(X, y, stump)
        prediction = predict(X, all_parameters, len(np.unique(y)))

        scatter = axes[idx].scatter(X[:, 0], X[:, 1],
                                    c=prediction,
                                    cmap='tab20')
        axes[idx].set_xlabel("X")
        axes[idx].set_ylabel("Y")
        axes[idx].set_title(f"Decision Stump with {stump} stumps")
        plt.colorbar(scatter, ax=axes[idx])
        print(f"Unique predictions for {stump} stumps:", np.unique(prediction))

    plt.tight_layout()
    plt.show()
def main():
    X_test = np.loadtxt('test.csv', delimiter=',', skiprows=1)[:, :-1]
    y_test = np.loadtxt('test.csv', delimiter=',', skiprows=1)[:, -1]
    X = np.loadtxt('train.csv', delimiter=',', skiprows=1)[:, :-1]
    y = np.loadtxt('train.csv', delimiter=',', skiprows=1)[:, -1]
    scatter_plot(X,y)

main()




