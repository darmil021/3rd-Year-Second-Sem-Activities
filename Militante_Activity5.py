# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:27:50 2024

@author: Dariel M. Militante
"""

import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

mu = np.linspace (1.65, 1.8, num = 50)
test= np.linspace (0,2)
uniform_dist = sts.uniform.pdf(mu) + 1 #sneaky advanced note: I'm using the uniform distribution for clarity,
                                       #but we can also make the beta distribution look completely flat by tweaking alpha and beta:
uniform_dist= uniform_dist/uniform_dist.sum() #Normalizing the distribution to make the probability densities sum into 1
beta_dist = sts.beta.pdf(mu, 2, 5, loc = 1.65, scale = 0.2)
beta_dist = beta_dist/beta_dist.sum()
plt.plot(mu, beta_dist, label = 'Beta Dist')
plt.plot(mu, uniform_dist, label = 'Uniform Dist')
plt.xlabel("Value of $\mu$ in meters")
plt.ylabel("Probability density")
plt.legend()

def likelihood_func(datum, mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale = 0.1) #Note that mu here is an array of values, so the output is also an array!
    return likelihood_out/likelihood_out.sum()

likelihood_out = likelihood_func(1.7, mu)
plt.plot(mu, likelihood_out)
plt.title("Likelihood of $\mu$ given observation 1.7m")
plt.ylabel("Probability Density/Likelihood")
plt.xlabel("Value of  $\mu$")
plt.show()

import scipy as sp

unnormalized_posterior = likelihood_out*uniform_dist
plt.plot(mu, unnormalized_posterior)
plt.xlabel("$\mu$ in meters")
plt.ylabel("Unnormalized Posterior")
plt.show()