import numpy as np

class LogisticRegression:

    def __init__(self, alpha=0.01, regLambda=0.01, epsilon=0.0001, maxNumIters=10000):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.theta = None

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        m = len(y)
        h = self.sigmoid(X @ theta)
        J = (-1/m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h)) + (regLambda / (2 * m)) * np.sum(theta[1:]**2)
        return J.item()

    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        m = len(y)
        h = self.sigmoid(X @ theta)
        grad = (1/m) * X.T @ (h - y) + (regLambda / m) * np.concatenate(([0], theta[1:]))
        return grad

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n, d = X.shape
        X = np.concatenate((np.ones((n, 1)), X), axis=1)  # Add a column of ones to X for the bias term
        self.theta = np.zeros((d + 1, 1))

        for _ in range(self.maxNumIters):
            gradient = self.computeGradient(self.theta, X, y, self.regLambda)
            self.theta -= self.alpha * gradient

            if np.linalg.norm(gradient) < self.epsilon:
                break

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        n = X.shape[0]
        X = np.concatenate((np.ones((n, 1)), X), axis=1)  # Add a column of ones to X for the bias term
        predictions = np.round(self.sigmoid(X @ self.theta))
        return predictions.flatten().astype(int)

    def sigmoid(self, Z):
        '''
        Computes the sigmoid function 1/(1+exp(-z))
        '''
        return 1 / (1 + np.exp(-Z))
