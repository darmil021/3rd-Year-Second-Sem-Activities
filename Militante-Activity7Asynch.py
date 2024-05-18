# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:04:35 2024

@author: Dariel M. Militante CAS-05-601A
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate some random data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term to input features
X_b = np.c_[np.ones((100, 1)), X]

# Define prior parameters
prior_mean = np.zeros((2, 1))
prior_covariance = np.eye(2)

# Define likelihood function
def likelihood(X, y, theta):
    sigma_squared = 1  # Variance of the noise
    m = len(y)
    residuals = y - X.dot(theta)
    return (1 / np.sqrt(2 * np.pi * sigma_squared) ** m) * np.exp(-0.5 * (1 / sigma_squared) * np.sum(residuals ** 2))

# Define posterior calculation
def posterior(X, y, prior_mean, prior_covariance):
    XTX_inv = np.linalg.inv(X.T.dot(X) + prior_covariance)
    theta_hat = XTX_inv.dot(X.T).dot(y)
    posterior_covariance = XTX_inv
    return theta_hat, posterior_covariance

# Perform MCMC sampling
def mcmc_sampling(X, y, prior_mean, prior_covariance, num_samples=1000):
    samples = []
    current_theta = np.random.randn(2, 1)  # Initialize randomly
    for _ in range(num_samples):
        proposed_theta = np.random.multivariate_normal(current_theta.flatten(), prior_covariance)
        prior_current = likelihood(X, y, current_theta)
        prior_proposed = likelihood(X, y, proposed_theta.reshape(-1, 1))
        acceptance_ratio = min(1, prior_proposed / prior_current)
        if np.random.rand() < acceptance_ratio:
            current_theta = proposed_theta.reshape(-1, 1)
        samples.append(current_theta)
    return np.array(samples)

# Perform Bayesian inference
theta_hat, posterior_covariance = posterior(X_b, y, prior_mean, prior_covariance)

# Perform MCMC sampling
samples = mcmc_sampling(X_b, y, prior_mean, prior_covariance)

# Plot the results
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, X_b.dot(theta_hat), color='red', label='Maximum Likelihood Estimate')
for sample in samples:
    plt.plot(X, X_b.dot(sample), color='green', alpha=0.01)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Bayesian Linear Regression')
plt.legend()
plt.show()
