# Conditional VAE in JAX

**WIP.**
This repository will (hopefully) contain an implementation of the *conditional variational autoencoder* (CVAE) introduced in [Learning Structured Output Representation using Deep Conditional Generative Models](https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html) in JAX. 

**Common questions.**

1. How is the Kullback-Leibler divergence term in the VAE loss function derived?

The approximate posterior distribution is assumed to be a multivariate Gaussian
distribution with diagonal covariance matrix. The prior distribution is assumed
to be a multivariate *standard* Gaussian distribution.

Recall that in general the Kullback-Leibler (KL) divergence between two
multivariate Gaussian distributions with mean vectors $\mu_1$ and $\mu_2$ and
covariance matrices $\Sigma_1$ and $\Sigma_2$ is given by

$$
\mathbb{D}_\text{KL}(
    \mathcal{N}(\mu_1, \Sigma_1) \mid\mid \mathcal{N}(\mu_2, \Sigma_2)
) =
\frac{1}{2} \left(
    \log \frac{
        \det \Sigma_2
    }{
        \det \Sigma_1
    }
    - n
    + \text{tr}(\Sigma_2^{-1} \Sigma_1)
    + (\mu_2 - \mu_1)^\top \Sigma_2^{-1} (\mu_2 - \mu_1)
\right).
$$

(see [here](https://stanford.edu/~jduchi/projects/general_notes.pdf) for detailed derivation).