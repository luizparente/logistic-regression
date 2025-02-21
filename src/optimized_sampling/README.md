# `BasicLeverageScoresSampler`: Leverage-Score-Based Sampling Algorithm Implementation

This sampling algorithm is used to approximate the logistic regression model by selecting a representative subset of the data, based on the leverage scores of the rows in the dataset.

## Implementation Overview

The goal of the algorithm is to efficiently approximate the full dataset using a sampled subset based on leverage scores. This is achieved by performing Singular Value Decomposition (SVD) on the data matrix $X$ to calculate the leverage scores, then using those scores to sample a subset of the data. Finally, the sampled data is used to train a Logistic Regression model, which is expected to approximate the performance of a model trained on the entire dataset.

### Leverage Score Calculation
The leverage scores are computed with method `compute_leverage_scores`, which uses **Singular Value Decomposition (SVD)**. It decomposes the matrix $X$ into its singular vectors and values. The leverage score for each dataset entry (row in the matrix) is calculated as the sum of the squares of the elements in the corresponding row of the left singular vector matrix $U$.

This step directly follows the approach proposed in the paper. Specifically, the paper states that leverage scores can be computed using **SVD** to capture the importance of each data point in the dataset with respect to the model. Even though the paper does not specify exactly how the leverage scores are used to calculate the final sample, it does mention that leverage scores help to sample data points that contribute the most to the model’s accuracy.

### Sampling Based on Leverage Scores
In method `sample`, the leverage scores are first normalized to create a probability distribution that sums to 1. This normalized distribution is then used to sample rows from the dataset $X$. The number of rows sampled is determined by the `sample_percentage` parameter, which controls how much of the dataset is selected.

After selecting the indices of the sampled rows, the corresponding rows in the feature matrix `X` and target vector `y` are extracted to form the sampled dataset.

This sampling process closely follows the paper’s method. The paper specifies that the **leverage scores** are used to form a probability distribution from which data points are selected. Rows with higher leverage scores are more likely to be chosen for the sample, ensuring that important data points (those that affect the model the most) are more likely to be included in the subsampled dataset.


## Differences Between This Implementation and the Paper's Approach

This implementation simplifies the strategy taken by the authors. It offers a simplified version of the approach described in the paper. While it is **more computationally efficient** and **easy to use**, it does not fully capture the **optimizations** and **error guarantees** of the paper's more complex sampling model. The paper's approach would be more appropriate for high-dimensional datasets or applications where precise error bounds and model accuracy are critical. However, for many practical purposes, the simpler implementation can provide a reasonable trade-off between speed and approximation accuracy.

More objecively, class `BasicLeverageScoresSampler` differs from the paper in the aspects described below.

### Sampling Process

The paper describes a more detailed approach where not only the leverage scores are used to sample the data, but also a **sketching matrix** is built, which allows the Logistic Regression model to be approximated more efficiently by reducing the dimensionality of the problem. The full matrix $S$ is used to modify how the data is projected into a lower-dimensional space, enabling faster training.

In contrast, our implementation here **samples rows based on leverage scores**, but skips the detailed step of constructing the sketching matrix $S$. This simplifies the implementation, but results in a slightly less optimized solution as a trade-off.

### Error Bound and Theoretical Guarantees

The paper provides theoretical guarantees for how well the sampled data approximates the full dataset’s Logistic Regression model. This involves complex mathematical analysis to bound the error in terms of approximation.

Our implementation does not include these guarantees. While it is expected to work well in practice, wedo not have formal error bounds for the approximation quality.

### Computation of Leverage Scores

The paper suggests using leverage scores to form a **probability distribution** that is then used to sample rows. The approximation is based on a **sketching matrix** and further optimizations.

Our implementation computes leverage scores in the same way, but does not take into account additional optimizations such as **oblivious sketching**, which the paper might use for better efficiency.

### Simplification Advantages

There are key advantages to simplifying the approach proposed in the paper:

- **Simplicity**: The implementation here is **simpler** and easier to understand. It directly samples rows based on leverage scores, which is computationally efficient and suitable for smaller datasets or less complex problems.
- **Speed**: By using only leverage score sampling without constructing a sketching matrix, the algorithm runs **faster** and requires **fewer computations**, making it suitable for situations where **speed** is more critical than exact accuracy.
  
### Simplification Disadvantages

Nevertheless, there is a trade-off between simplicity and accuracy:

- **Less Accurate**: Because the implementation skips the step of constructing the sketching matrix, the approximation may not be as accurate as the one described in the paper. Without the full matrix, the approximation quality could be compromised, especially for large, high-dimensional datasets.
- **Lack of Theoretical Guarantees**: Unlike the paper, this implementation does not provide formal guarantees on the error bounds, which might be important for critical applications.