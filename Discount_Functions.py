import pandas as pd
import numpy as np 
from scipy.stats import norm
from numpy.linalg import inv

# DIY OSL function
## takes a matrix X and a vector y and can calculate beta = inv(X'X)*(X'y), as well as y_hat = X*beta
class OLS:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.beta = np.dot(inv(np.dot(pd.DataFrame.transpose(X),X)), np.dot(pd.DataFrame.transpose(X),y))
        self.reg_y = np.dot(X,self.beta)

# Tailored function for 203M A1Q1 to generate the basis vector
def power_5(data):
    return pd.DataFrame({'T' :data,
                        'T^2': np.power(data,2),
                        'T^3': np.power(data,3),
                        'T^4': np.power(data,4),
                        'T^5': np.power(data,5) })

def poly5_to_Z(z_OLS, T):
    ## This function takes maturity and a fitted ploynomial interest rate model and return Z(0,T)
    ## z_OLS must be a OLS class and T must be a list
    return np.exp(np.dot(power_5(T), z_OLS.beta))

## Black formula    
def Black_formula(z_fun, T, r_strike, delta, vol, t = 0):
    ## z_fun is a function of two argument that takes maturity and output Z. i.e Z(t,T) = z_fun(t,T)
    ## or a function of one argument Z(T-t)
    ## T is maturity time, t is current time
    ## r_strike is the strike
    ## delta is the time difference betwwen t_k=1 - t_k
    ## vol is sig_f
    z_tkplus1 = z_fun(T-t+delta)
    z_tk = z_fun(T-t)
    forward = -1/delta*np.log(z_tkplus1/z_tk)
    d1 = (np.log(forward/r_strike) + np.power(vol,2)/2*(T-t))/(vol*np.sqrt(T-t))
    d2 = d1 - vol*np.sqrt(T-t)
    cap_price = z_tkplus1 * delta *(forward*norm.cdf(d1) - r_strike*norm.cdf(d2))
    return cap_price