# Course Notes Structure

## Foundations
- An Introduction to AI
    - Machine Learning: Overview and Motivation
    - Learning from data
    - Prediction vs Inference vs Decision Making
- Definitions and Notation
    - Data, features, labels
    - Supervised vs unsupervised vs semi-supervised
    - Classification vs regression
    - Deterministic vs probabilistic models
    - Parameters, hyperparameters, loss functions
- Probability Theory
    - Random variables
    - Distributions
    - Expectation, variance
    - Bayes’ rule
- Statistical Inference
    - Estimation (MLE, MAP)
    - Bias – Variance
    - Confidence intervals and uncertainty
- Linear Algebra
    - Vectors, matrices
    - Matrix multiplication
    - Linear transformations
    - Eigenvalues, Eigenvectors and SVD
- Calculus and Optimisation Basics
    - Gradients
    - Derivatives of common functions
    - Convexity
    - A first look at Gradient Descent

## Data
- Data Types and Structures
- Exploratory Data Analysis (EDA)
- Data Cleaning and Preprocessing
    - Missing values
    - Normalisation and standardisation
    - Encoding categorical variables
    - Train/validation/test splits
- Feature Engineering and Selection
- Dimensionality Reduction (PCA introduced here conceptually)

## Statistical Learning Theory
- Learning as Function Approximation
- Loss Functions and Risk Minimisation
- Empirical Risk Minimisation
- Generalisation, Overfitting, and Underfitting
- Regularisation (L1, L2)
- Bias–Variance Revisited with Examples

## Supervised Learning 
- Supervised Learning Framework
- Linear Regression
    - Closed-form solution
    - Gradient-based optimisation
    - Regularised regression (Ridge, Lasso)
- Classification
    - Logistic regression
    - Softmax regression
- Evaluation Metrics
    - Accuracy, precision, recall, F1, ROC, AUC
    - Regression metrics (MSE, MAE, R2)
- Model Selection and Validation
    - Cross-validation
    - Hyperparameter tuning
    - Learning curves

## Unsupervised Learning
- Clustering
    - K-means
    - Hierarchical clustering
    - Density-based methods (DBSCAN)
- Dimensionality Reduction
    - PCA
    - Introduction to Kernel PCA
- Mixture Models
    - Gaussian Mixture Models
    - Expectation–Maximisation (EM algorithm)

## Semi-Supervised Learning
- Motivating Limited Label Scenarios
- Self-training
- Consistency Regularisation
- Graph-based Semi-supervised Methods

## Learning Parametric Models
- Parametric vs Nonparametric Models
- Maximum Likelihood in Parametric Settings
- Exponential Family Models
- Generalised Linear Models

## Artificial Neural Networks
- The Perceptron
- Multi-layer Perceptrons (MLPs)
- Activation Functions
- Backpropagation (Derivation and Examples)
- Optimisation Algorithms (SGD, Momentum, Adam)
- Regularisation in Neural Nets (Dropout, Weight Decay)

## Deep Learning
- Representation Learning
- Convolutional Neural Networks
- Recurrent Networks & Sequence Models
- Modern Architectures (Transformers - conceptual)
- Training Dynamics and Practical Tricks

## Ensemble Methods
- Bagging and Bootstrap Aggregation
- Random Forests
- Boosting
    - AdaBoost
    - Gradient Boosting
    - XGBoost / LightGBM (conceptual mechanics)

## Kernel Methods
- The Kernel Trick
- Support Vector Machines
- Kernel Ridge Regression
- Kernel PCA (full revisit)

## Gaussian Processes
- Nonparametric Bayesian Modelling
- Covariance Functions and Kernels
- GP Regression
- GP Classification

## Generative Models
- Generative vs Discriminative Models
- Latent Variable Models
- Variational Autoencoders
- GANs
- Normalising Flows

## Performance
- Model Evaluation Revisited: Robustness & Stability
- Calibration
- Fairness, Bias, Ethical Considerations
- Computational Efficiency (Floating point errors)