# Logistic Regression Implementation

This repository implements the classic Logistic Regression machine learning algorithm. Additionally, we explore an optimization as proposed by Agniva Chowdhury and Pradeep Ramuhalli in [A Provably Accurate Randomized Sampling Algorithm for Logistic Regression](https://ojs.aaai.org/index.php/AAAI/article/view/29042).

The Logistic Regression implementation presented here is intentionally not optimized. The rationale for this decision is that a slower model would allow for better measurement and visualization of the impacts caused by the sampling algorithms explored here, which are the focus of this work.

All code presented here is restricted to native Python functions and NumPy.

## How this Repository is Structured

The code proposed here is organized as follows:

- Directory `src` is the root and contains the source code for this project.
- Sub-directory `experiments` provides our experiments designed to benchmark the effectiveness of the randomized sampling algorithms studied in this work.
- Sub-directory `sampling` provides a simplified implementation of the optimized sampling algorithm proposed in the aforementioned paper, as well as a random sampler for comparison.
- Sub-directory `regression` provides our "vanilla" implementation for the Logistic Regression algorithm.
- Sub-directory `utilities` implements basic utilities for obtaining additional performance metrics for the models.

```
src
├── experiments
├── sampling
├── regression
└── utilities
```

Further documentation on the particular implementations can be found in their corresponding directories.