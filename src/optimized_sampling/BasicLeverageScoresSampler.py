import numpy as np

class BasicLeverageScoresSampler:
    def __init__(self):
        pass

    def compute_leverage_scores(self, X):
        """
        This method calculates the leverage scores using Singular Value Decomposition (SVD).
        
        Parameters:
        - X: A 2D numpy array (n_samples x n_features)
        
        Returns:
        - leverage_scores: A 1D numpy array with the leverage scores for each row in X.
        """
        # Performing Singular Value Decomposition (SVD).
        left_singular_vectors, _, _ = np.linalg.svd(X, full_matrices=False)
        
        # Calculating scores = squared norms of the left singular vectors.
        leverage_scores = np.sum(left_singular_vectors**2, axis=1)

        return leverage_scores
    
    def sample(self, X, y, sample_percentage):
        """
        This method samples rows from the dataset X based on computed leverage scores, which are calculated from the SVD of X.
        Returns the sampled feature matrix X_sampled and target vector y_sampled.

        Parameters:
        - X: A 2D numpy array (n_samples x n_features).
        - y: A 1D numpy array with the target labels.
        - sample_percentage: A decimal value (e.g., 0.2 for 20%) indicating the percentage of rows to sample from the original dataset.

        Returns:
        - X_sampled: A 2D numpy array with the sampled rows from X.
        - y_sampled: A 1D numpy array with the sampled rows from y.
        """
        
        # Computing leverage scores.
        leverage_scores = self.compute_leverage_scores(X)
        
        # Normalizing leverage scores in ordr to create a probability distribution.
        normalized_probabilities = leverage_scores / np.sum(leverage_scores)
        
        # Calculating the number of samples to draw based on the sample_percentage.
        num_samples = int(sample_percentage * X.shape[0])
        
        # Sampling indices based on the normalized leverage scores.
        sampled_indices = np.random.choice(X.shape[0], size=num_samples, p=normalized_probabilities)
        
        # Extracting the sampled data.
        X_sampled = X[sampled_indices]
        y_sampled = y[sampled_indices]
        
        return X_sampled, y_sampled