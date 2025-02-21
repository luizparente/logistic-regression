# Logistic Regression Implementation

This repository implements the classic Logistic Regression machine learning algorithm. Additionally, we explore an optimization as proposed by Agniva Chowdhury and Pradeep Ramuhalli in [A Provably Accurate Randomized Sampling Algorithm for Logistic Regression](https://ojs.aaai.org/index.php/AAAI/article/view/29042).

All code presented here is restricted to native Python functions and NumPy.

## How this Repository is Structured

The code proposed here is organized as follows:

- Directory `src` contains the source code for this project.
   - `requirements.txt` lists the requirements for the Python virtual environment necessary for the project.
   - `model_benchmarking.ipynb` provides tests and comparisons for our `LogisticRegression` versus other implementations and variations.
- Sub-directory `ours` provides our "vanilla" implementation for the Logistic Regression algorithm, as detailed below.
- Sub-directory `optimized_sampling` provides a simplified implementation of the optimized sampling algorithm proposed in the aforementioned paper.
- Sub-directory `utilities` implements basic utilities for obtaining additional performance metrics for the models.

```
src
├── ours
│   └── LogisticRegression.py
├── optimized_sampling
│   └── BasicOptimizedSampler.py
├── utilities
│   └── Stopwatch.py
├── requirements.txt
└── model_benchmarking.ipynb
```

Further documentation on the particular implementations can be found in their corresponding directories.