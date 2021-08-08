import numpy as np


class NaiveBayes:

    def fit(self, X: np.ndarray, y: np.array):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # initialise mean, variance and priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[c == y]
            self._mean[idx,:] = X_c.mean(axis=0)
            self._var[idx,:] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X: np.ndarray) -> np.array:
        return [self._predict(x) for x in X]

    def _predict(self, x: np.array) -> float:
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(
                np.log(self._pdf_gaussian(idx,x))
            )
            posteriors.append(prior + class_conditional)
        
        return self._classes[np.argmax(posteriors)]

    def _pdf_gaussian(self, class_idx: int, x: np.array) -> np.array:
        mean = self._mean[class_idx]
        var = self._var[class_idx]

        numerator = np.exp((-(x-mean)**2) / (2*var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator/denominator
