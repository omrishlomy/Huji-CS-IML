
import numpy as np
import faiss
import matplotlib.pyplot as plt


testing_data = np.loadtxt('test.csv',delimiter=',',skiprows=1)[:,:-1]
testing_labels = np.loadtxt('test.csv',delimiter=',',skiprows=1)[:,-1]
training_data = np.loadtxt('train.csv',delimiter=',',skiprows=1)[:,:-1]
training_labels = np.loadtxt('train.csv',delimiter=',',skiprows=1)[:,-1]

class KNNClassifier:
    def __init__(self, k, distance_metric='l2'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.Y_train = None

    def fit(self, X_train, Y_train):
        """
        Update the kNN classifier with the provided training data.

        Parameters:
        - X_train (numpy array) of size (N, d): Training feature vectors.
        - Y_train (numpy array) of size (N,): Corresponding class labels.
        """
        self.X_train = X_train.astype(np.float32)
        self.Y_train = Y_train
        d = self.X_train.shape[1]
        if self.distance_metric == 'l2':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L2)
        elif self.distance_metric == 'l1':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L1)
        else:
            raise NotImplementedError
        self.index.add(self.X_train)

    def predict(self, X):
        """
        Predict the class labels for the given data.

        Parameters:
        - X (numpy array) of size (M, d): Feature vectors.

        Returns:
        - (numpy array) of size (M,): Predicted class labels.
        """
        X = X.astype(np.float32)
        d,i = self.index.search(X, self.k)
        indexes = self.Y_train[i]
        def mode(index):
            unique_val,count = np.unique(index, return_counts=True)
            modes = count.argmax()
            return unique_val[modes]
        return np.apply_along_axis(mode,1,indexes)




    def knn_distance(self, X):
        """
        Calculate kNN distances for the given data. You must use the faiss library to compute the distances.
        See lecture slides and https://github.com/facebookresearch/faiss/wiki/Getting-started#in-python-2 for more information.

        Parameters:
        - X (numpy array) of size (M, d): Feature vectors.

        Returns:
        - (numpy array) of size (M, k): kNN distances.
        - (numpy array) of size (M, k): Indices of kNNs.
        """
        X = X.astype(np.float32)
        d,i = self.index.search(X, self.k)
        return d,i

def all_k_and_l():
    k_list = [1,10,100,1000,3000]
    l_list = ['l1','l2']
    for k in k_list:
        k_accuracy = []
        for l in l_list:
            KNNClass = KNNClassifier(k, l)
            KNNClass.fit(training_data, training_labels)
            predictions = KNNClass.predict(testing_data)
            accuracy = np.mean(predictions == testing_labels)
            k_accuracy.append(accuracy)
        print(f'k = {k}, l1 accuracy = {k_accuracy[0]}, l2 accuracy = {k_accuracy[1]}')
def anomaly_detection(test_set,P=50):
    KNNClass = KNNClassifier(5, 'l2')
    KNNClass.fit(training_data,training_labels)
    distances,indexes = KNNClass.knn_distance(test_set)
    sums_score = np.apply_along_axis(np.sum,1,distances)
    anomaly_scores = np.argpartition(sums_score,-P)[-P:]
    sorted_indices = anomaly_scores[np.argsort(sums_score[anomaly_scores])[::-1]]
    return anomaly_scores,sorted_indices


def plot_anomaly(indexes, scores, test_set):
    # Plot all points in test_set
    plt.scatter(training_data[:, 0],training_data[:, 1], c='black', alpha=0.01)

    # Create colors array for highlighting anomalies
    colors = ['red' if i in indexes else 'blue' for i in range(len(test_set))]

    # Plot points with anomalies highlighted
    plt.scatter(test_set[:, 0], test_set[:, 1], c=colors, alpha=0.5)

    plt.title('Anomaly Detection Results')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# anomaly_set = np.loadtxt('AD_test.csv',delimiter=',',skiprows=1)
# scores,indexes = anomaly_detection(anomaly_set)
# print(anomaly_set.shape)
# plot_anomaly(indexes,scores,test_set=anomaly_set)
# all_k_and_l()
