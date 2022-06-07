# Linear Regression Simulation and parameter MLE

# package imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize # as maximize does not exist
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Generate 'random' data
np.random.seed(0)
x = 1.5 + 2.5 * np.random.randn(100)   # Independent variable - 100 values from a normal distribution
                                       # with Mean = 1.5 and SD = 2.5
res = 0.5 * np.random.randn(100)       # Generate 100 residual terms with Mean = 0 and SD = 0.5
y = 2 + 0.3 * x + res                  # Dependent variable; parameters fixed

# plot data
fig, ax = plt.subplots()
ax.plot(x, y, 'b.')
plt.show()

# define likelihood function
def lik(parameters, x, y): 
    w0 = parameters[0] 
    w1 = parameters[1] 
    sigma = parameters[2] 
    
    y_hat = w0 + w1 * x
        
    L = np.sum(np.log(stats.norm.pdf(y - y_hat, loc = 0, scale=sigma)))
    return -L

# pre-requisites
def constraints(parameters):
    sigma = parameters[2]
    return sigma

cons = {
    'type': 'ineq',
    'fun': constraints
}

# run MLE
lik_model = minimize(lik, np.array([2, 2, 2]), args=(x,y,), constraints=cons)

# get the parameters from MLE
mle_parameters = [lik_model.x[0], lik_model.x[1], lik_model.x[2]]
print(mle_parameters)

# plot the regression line and data
fig, ax = plt.subplots()
ax.plot(x, y, 'b.')
xx = np.linspace(np.min(x), np.max(x), 100)
yy = lik_model.x[0]  +  lik_model.x[1] * xx
ax.plot(xx, yy, 'r-')                                                 
plt.show()

# same parameters via OLS as well
x_new = pd.DataFrame([np.ones(100), x]).transpose()
model = sm.OLS(pd.Series(y), x_new) 
model_fit = model.fit() 
print(model_fit.summary())
np.sqrt(model_fit.scale)

