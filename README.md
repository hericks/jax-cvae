# Conditional VAE in JAX

**WIP.**
This repository will (hopefully) contain an implementation of the *conditional variational autoencoder* (CVAE) introduced in [Learning Structured Output Representation using Deep Conditional Generative Models](https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html) in JAX. 


**Setup.**

We model the MNIST images as samples from a generative model $p_{\boldsymbol \theta}(\mathbf{x}, \mathbf{z}) = p_{\boldsymbol \theta}(\mathbf{z}) p_{\boldsymbol \theta}(\mathbf{x} \mid \mathbf{z})$.

Following Section 3 of [1], we let
- the prior over the latent variables be the centered isotropic multivariate Gaussian $p_{\boldsymbol \theta}(\mathbf{z}) = \mathcal{N}(\mathbf{z}; \mathbf{0}, \mathbf{I})$.
Especially, there are no unknown model parameters in the prior.
- $p_{\boldsymbol \theta}(\mathbf{x} \mid \mathbf{z})$ be multivariate Gaussian with mean $\boldsymbol \mu_{\boldsymbol \theta}(\mathbf{z})$ and diagonal covariance matrix $\boldsymbol \sigma^2_{\boldsymbol \theta}(\mathbf{z})$. The distribution parameters are computed from $\mathbf{z}$ using a neural network with parameters $\boldsymbol \theta$.


**Common questions.**

1. How is the Kullback-Leibler divergence term in the VAE loss function derived?

    The VAE loss function contains a term for the Kullback-Leibler (KL) divergence between the approximate posterior
    $q_\phi(z \mid x^{(i)})$ and the prior $p_\theta(z)$.

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

## References
[1] D. Kingma and M. Welling.
Auto-Encoding Variational Bayes.
2nd International Conference on Learning Representations (ICLR 2014), Banff, AB, Canada.