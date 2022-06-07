"""Linear regression example.

see: https://docs.pymc.io/en/v3/pymc-examples/examples/getting_started.html
If you run into problems with your configuration (fix it or;-)
run the code on google colab
"""

# Imports and config
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm


def main():
    print(f'Running on PyMC3 v{pm.__version__}')

    # Initialize random number generator
    RANDOM_SEED = 8927
    np.random.seed(RANDOM_SEED)
    az.style.use("arviz-darkgrid")

    # True parameter values
    alpha, sigma = 1, 1
    beta = [1, 2.5]

    # Size of dataset
    size = 100

    # Predictor variable
    X1 = np.random.randn(size)
    X2 = np.random.randn(size) * 0.2

    # Simulate outcome variable
    Y = alpha + beta[0] * X1 + beta[1] * X2 + np.random.randn(size) * sigma

    # Plot
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
    axes[0].scatter(X1, Y, alpha=0.6)
    axes[1].scatter(X2, Y, alpha=0.6)
    axes[0].set_ylabel("Y")
    axes[0].set_xlabel("X1")
    axes[1].set_xlabel("X2")
    plt.show()

    # Modeling
    basic_model = pm.Model()

    with basic_model:

        # Priors for unknown model parameters
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Expected value of outcome
        mu = alpha + beta[0] * X1 + beta[1] * X2

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

    """
    The maximum a posteriori (MAP) estimate for a model, is the mode of the 
    posterior distribution and is generally found using numerical optimization 
    methods. This is often fast and easy to do, but only gives a point estimate 
    for the parameters and can be biased if the mode isn’t representative of the 
    distribution. PyMC3 provides this functionality with the find_MAP function."""

    map_estimate = pm.find_MAP(model=basic_model)
    print(f'Maximum a posterior: \n{map_estimate}')

    # MCMC sampling
    with basic_model:
        # draw 500 posterior samples
        trace = pm.sample(100, return_inferencedata=False)

    # Plot traces
    with basic_model:
        az.plot_trace(trace)

    plt.show()


if __name__ == '__main__':
    main()