# `SlowLogisticRegression` Class Implementation

This class provides an abstraction that allows users to create a Logistic Regression model instance that can be trained and used for predictions. The strategy employed for its implementation is described in the sub-sections below.

## Training Data

The model expects an input of training data in the following format:

- **Features (`X`):** A matrix of input features where each row represents a data point and each column represents a feature (e.g., a matrix of shape $n \times m$, where $n$ is the number of data points and $m$ is the number of features).

- **Labels (`y`):** A vector of length $n$ with the target binary labels (0 or 1, for binary logistic regression), corresponding to the input features.

## Initialization

Model parameters are initialized as follows:

- **Weights (`θ`):** The weights (coefficients) of the model, here initialized as zeros. These weights are of size $m$ (one for each feature).

- **Bias (`b`):** A scalar value added to the output of the linear combination of the features, here initialized to zero.

## Model Hypothesis

A linear combination (linear model) is implemented for the inputs and weights.

- **Linear Combination:** For each data point, we compute the weighted sum of the features plus the bias term. 

$$
z = \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_m x_m + b
$$

- **Sigmoid Function (Logistic Function):** This function maps the output of the linear combination to a probability between 0 and 1, which represents the probability of the positive class (prediction being `true`). We apply the sigmoid function to the linear combination $z$ to obtain the predicted probability:

$$
\hat{y}(z) = \frac{1}{1 + e^{-z}}
$$

## Loss Function

The loss function used in logistic regression is the **binary cross-entropy**, which measures the difference between the predicted probabilities versus the actual (expected) values. For a dataset of $n$ examples, the cost function $J(\theta)$ is given by:

$$
J(\theta) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
$$

Where $y^{(i)}$ is the true label and $\hat{y}^{(i)}$ is the predicted probability for each data point.

## Optimization

The goal of Logistic Regression is to find the optimal values for the weights and bias that minimize the loss function. This is typically done using an optimization algorithm like **gradient descent**, which goes as follows:

1. Compute the gradients (partial derivatives) of the loss function with respect to each parameter (weights and bias):

$$
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{n} \sum_{i=1}^{n} \left( \hat{y}^{(i)} - y^{(i)} \right) x_j^{(i)}
$$
$$
\frac{\partial J(\theta)}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} \left( \hat{y}^{(i)} - y^{(i)} \right)
$$

2. Update the weights and bias using the gradients, ($\alpha$ is the learning rate, which controls the step size of each update):

$$
\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
$$
$$
b := b - \alpha \frac{\partial J(\theta)}{\partial b}
$$

3. Repeat until the loss converges (i.e., the change in the cost function between iterations is small enough) or a predefined number of iterations is reached.

## Model Evaluation

We evaluate the model’s performance by comparing the predicted labels against the actual values. Common metrics include accuracy, precision, recall, F1 score, etc.

## Output

After training, the learned weights and bias can be used to make predictions on new data. The output is a probability, but for classification, a *Decision Boundary* (commonly 0.5) may be applied to convert the probability into a binary class label:

$$
\hat{y} = \begin{cases} 
1 & \text{if } \hat{y} \geq 0.5 \\
0 & \text{if } \hat{y} < 0.5 
\end{cases}
$$

## Example Usage

The snippet below illustrates the typical usage of the `SlowLogisticRegression` class using the [breast cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html). 

```
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from SlowLogisticRegression import SlowLogisticRegression

# Loading the breast cancer dataset.
data = load_breast_cancer()
X, y = data.data, data.target

# Splitting the data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features.
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Initializing and training the model.
logistic_regression_model = SlowLogisticRegression(learning_rate=0.1, epochs=5000)
logistic_regression_model.fit(X_train, y_train)

# Making predictions on the test set.
predictions = logistic_regression_model.predict(X_test)

# Evaluating model.
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy on test set: {accuracy * 100:.2f}%")

# Output: Accuracy on test set: 97.37%
```