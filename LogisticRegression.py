import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, linear_combination):
        """This method implements the Sigmoid activation function."""
        result = 1 / (1 + np.exp(-linear_combination))

        return result

    def compute_loss(self, feature_matrix, labels):
        """This method computes the binary cross-entropy loss."""
        num_samples = len(labels)
        predictions = self.sigmoid(np.dot(feature_matrix, self.weights) + self.bias)
        loss = - (1/num_samples) * np.sum(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))
        
        return loss

    def gradient_descent(self, feature_matrix, labels):
        """This method performs gradient descent to minimize the loss."""
        num_samples = len(labels)
        
        for epoch in range(self.epochs):
            predictions = self.sigmoid(np.dot(feature_matrix, self.weights) + self.bias)

            # Calculating gradients.
            gradient_weights = (1/num_samples) * np.dot(feature_matrix.T, (predictions - labels))
            gradient_bias = (1/num_samples) * np.sum(predictions - labels)

            # Updating weights and bias.
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

            # if epoch % 100 == 0:
            #     loss = self.compute_loss(feature_matrix, labels)
            #     print(f"Epoch {epoch}, Loss: {loss}")

    def fit(self, feature_matrix, labels):
        """This method trains the logistic regression model."""
        num_samples, num_features = feature_matrix.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Training the model using gradient descent.
        self.gradient_descent(feature_matrix, labels)

    def predict(self, feature_matrix):
        """This method predicts the binary vector output based on learned weights and bias."""
        linear_combination = np.dot(feature_matrix, self.weights) + self.bias
        predictions = self.sigmoid(linear_combination)
        
        result = [1 if probability >= 0.5 else 0 for probability in predictions]

        return result
