from collections import Counter
import numpy as np
from scipy.spatial import distance

class KNN:

    def __init__(self, k: int):
        """Initialize the KNN

        Args:
            k (int): number of clusters
        """
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to the training data/labels

        Args:
            X (np.ndarray): training data
            y (np.ndarray): training class labels
        """        
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class of a set of data

        Args:
            X (np.ndarray): Datapoints to predict class labels for

        Returns:
            np.ndarray: The predicted class labels
        """        
        return np.array([self._predict(x) for x in X])
    
    def _predict(self, x: np.ndarray) -> int:
        """Helper method to predict class labels for samples X

        Args:
            x (np.ndarray): Single sample to predict class label for

        Returns:
            int: predicted class label
        """        
        # p=2 minkowski metric is euclidian distance
        distances = [distance.minkowski(x, x_train, p=2) for x_train in self.X_train]
        # get k nearest neighbors and return the majority class vote
        k_neighbor_labels = [self.y_train[i] for i in np.argsort(distances)[: self.k]]
        return Counter(k_neighbor_labels).most_common(1)[0][0]
