import numpy as np
import torch
from torch import nn



class Ridge_Regression:

    def __init__(self, lambd):
        self.lambd = lambd
        self.W = None

    def fit(self, X, Y):

        """
        Fit the ridge regression model to the provided data.
        :param X: The training features.
        :param Y: The training labels.
        """
        Y = 2 * (Y - 0.5)  # transform the labels to -1 and 1, instead of 0 and 1.
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        correllation_matrix = np.dot(X.T, X) + (self.lambd * np.eye(X.shape[1]))
        prediction_vector = np.dot(X.T, Y)
        self.W = np.linalg.inv(correllation_matrix).dot(prediction_vector)
        # compute the ridge regression weights using the formula from class / exercise.
        # you may not use np.linalg.solve, but you may use np.linalg.inv

    def predict(self, X):
        """
        Predict the output for the provided data.
        :param X: The data to predict. np.ndarray of shape (N, D).
        :return: The predicted output. np.ndarray of shape (N,), of 0s and 1s.
        """
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        preds = np.zeros(X.shape[0])
        preds = np.dot(X, self.W)
        preds = (preds > 0).astype(int)


        ########## YOUR CODE HERE ##########

        # compute the predicted output of the model.
        # name your predicitons array preds.

        ####################################

        # transform the labels to 0s and 1s, instead of -1s and 1s.
        # You may remove this line if your code already outputs 0s and 1s.

        return preds



class Logistic_Regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Logistic_Regression, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim)


        ########## YOUR CODE HERE ##########

        # define a linear operation.

        ####################################
        pass

    def forward(self, x):
        """
        Computes the output of the linear operator.
        :param x: The input to the linear operator.
        :return: The transformed input.
        """
        # compute the output of the linear operator

        return self.linear(x)

        # return the transformed input.
        # first perform the linear operation
        # should be a single line of code.

        ####################################

        pass

    def predict(self, x):
        """
        THIS FUNCTION IS NOT NEEDED FOR PYTORCH. JUST FOR OUR VISUALIZATION
        """
        x = torch.from_numpy(x).float().to(self.linear.weight.data.device)
        x = self.forward(x)
        x = nn.functional.softmax(x, dim=1)
        x = x.detach().cpu().numpy()
        x = np.argmax(x, axis=1)
        return x
