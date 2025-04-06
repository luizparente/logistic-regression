import numpy as np

class RandomSampler:
    def __init__(self):
        pass

    def sample(self, X, y, sample_percentage):
        """
        This method now performs a truly uniform random sampling of rows from X and y.
        
        Parameters:
        - X: A 2D numpy array (n_samples x n_features).
        - y: A 1D numpy array with the target labels.
        - sample_percentage: A decimal value (e.g., 0.2 for 20%) indicating 
          the percentage of rows to sample from the original dataset.

        Returns:
        - X_sampled: A 2D numpy array with the sampled rows from X.
        - y_sampled: A 1D numpy array with the sampled rows from y.
        """

        # Calculate the number of samples to draw
        num_samples = int(sample_percentage * X.shape[0])

        # Draw a random sample of indices (without replacement)
        sampled_indices = np.random.choice(
            X.shape[0],
            size=num_samples,
            replace=False
        )

        # Extract the sampled data
        X_sampled = X[sampled_indices]
        y_sampled = y[sampled_indices]

        return X_sampled, y_sampled
