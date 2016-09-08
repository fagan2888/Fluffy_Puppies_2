from __future__ import print_function, division ##python 3 print function
import numpy as np
import pandas as pd
import scipy as sp
import math
from scipy.stats import norm
from numpy.linalg import inv
import matplotlib.pyplot as plt
from operator import add


## Hull-White caplet price
def insta_f(t, coeff):
    a,b,c,d,e = coeff
    return -(a + 2*b*t + 3*c*t**2 + 4*d*t**3 + 5*e*t**4)

def insta_f_derivative(t, coeff):
    a,b,c,d,e = coeff
    return -(2*b + 6*c*t + 12*d*t**2 + 20*e*t**3)

def calibrate_HW(kappa,sigma):
    HW_cap_price = 0
    for i in range(3,40):
        HW_cap_price += (1+r_cap*delta) * HW.put(0,data_z.loc[i,'Maturity'],data_z.loc[i-1,'Maturity'],K,r0,kappa,sigma)
    return HW_cap_price

def obj_func(x):
    kappa = x[0]
    sigma = x[1]
    return (calibrate_HW(kappa,sigma) - cap_price)**2

    
class Hull_White:
    def __init__(self):
        return

    def theta(self,t,kappa,sigma, coeff):
        return insta_f_derivative(t, coeff) + kappa * insta_f(t, coeff) + sigma**2/(2*kappa)*(1-np.exp(-2*kappa*t))
    
    def B(self,t,T,kappa,sigma):
        return (1-np.exp(-kappa*(T-t)))/kappa
    
    def A(self,t,T,kappa,sigma, coeff):
        return -(sp.integrate.quad(lambda tau: self.B(tau,T,kappa,sigma)*self.theta(tau,kappa,sigma, coeff), t,T))[0] + \
               sigma**2/(2*kappa**2)*(T-t+(1-np.exp(-2*kappa*(T-t)))/(2*kappa)-2*self.B(t,T,kappa,sigma))
    
    def Z(self,t,T,r,kappa,sigma, coeff):
        return np.exp(self.A(t,T,kappa,sigma, coeff) - self.B(t,T,kappa,sigma)*r)
    
    def r_t(self, t, T, r, kappa, sigma, coeff):
        return -self.A(t,T,kappa,sigma, coeff)/(T-t) + \
        (1.0/kappa)*((1-np.exp(-kappa*(T-t)))/(T-t))*r
    
    def put(self,t,T,s,K,r,kappa,sigma, coeff):
        sigma_z = np.sqrt(sigma**2/(2*kappa**3) * (1-np.exp(-2*kappa*(T-t))) * (1-np.exp(-kappa*(s-T)))**2)
        d1 = np.log(self.Z(t,s,r,kappa,sigma, coeff)/(K * self.Z(t,T,r,kappa,sigma, coeff)))/sigma_z + sigma_z/2
        d2 = d1 - sigma_z
        return K*self.Z(t,T,r,kappa,sigma, coeff)*norm.cdf(-d2) - self.Z(t,s,r,kappa,sigma, coeff)*norm.cdf(-d1)
        
    def Plot_Theta(self, t_range, kappa, sigma, coeff):
        theta_arr = []
        for t in t_range:
            theta_arr.append(self.theta(t, kappa, sigma, coeff))
        plt.plot(t_range, theta_arr)
        plt.title('Monthly theta(t)')
        plt.xlabel('time (years)')
        plt.ylabel('theta(t)')
        plt.savefig('theta(t) plot')
        return theta_arr
     
    def Monte_Carlo_2(self, kappa, sigma, r0, T, dt, coeff, num_sims):
        np.random.seed(0)
        # need to account for prepayment spped of 150%
        iterations = int(T / dt)
        
        theta_arr = [self.theta(dt*i, kappa, sigma, coeff) for i in range(iterations)]
        df_matrix = np.zeros((num_sims, iterations))
        df_anti_matrix = np.zeros((num_sims, iterations))
        r_matrix = np.zeros((num_sims, iterations))
        r_anti_matrix = np.zeros((num_sims, iterations))
        for j in range(num_sims):
            norm_arr = np.random.normal(0,1,size=iterations)    # normal random vars
            anti_arr = [-x for x in norm_arr]                   # antithetic norm random vars
            df_arr = []
            df_anti_arr = []
            r, r_anti = [r0]*2
            r_arr = []
            r_anti_arr = []
        
            for i in range(iterations):
                delta_r = (theta_arr[i] - kappa * r) * dt + \
                           sigma*(math.sqrt(dt)*norm_arr[i])

                anti_delta_r = (theta_arr[i] - kappa * r_anti) * dt + \
                                sigma*(math.sqrt(dt)*anti_arr[i])
                r += delta_r
                r_anti += anti_delta_r
                
                
                r_arr.append(r)
                r_anti_arr.append(r_anti)
                
                df_arr.append(np.exp(-r*dt))
                df_anti_arr.append(np.exp(-r_anti*dt))
            
            r_matrix[j,:] = r_arr
            r_anti_matrix[j,:] = r_anti_arr
            df_matrix[j,:] = df_arr
            df_anti_matrix[j,:] = df_anti_arr

        # cumulative discount factor
        cum_df_matrix = np.zeros((num_sims, iterations))
        cum_df_anti_matrix = np.zeros((num_sims, iterations))
        
        for i in range(num_sims):
            cum_df_matrix[i,:] = df_matrix[i,:].cumprod()
            cum_df_anti_matrix[i,:] = df_anti_matrix[i,:].cumprod()
        return (cum_df_matrix, cum_df_anti_matrix, r_matrix, r_anti_matrix)

    def Std_Err(self, prices, num_sims):
        std_err_arr = []
        for col in prices.T:
            std_err_arr.append(float('%.3f' % (np.std(col) / math.sqrt(num_sims))))

        return std_err_arr

    def Effective_Duration(self, V0, V_plus, V_minus, dr):
        denom = np.asarray(V0) * 2.0 * dr
        numer = np.subtract(V_minus,V_plus)
        return np.divide(numer, denom)

    def Effective_Convexity(self, V0, V_plus, V_minus, dr):
        denom = np.asarray(V0) * dr**2
        numer = np.add(V_plus, V_minus)
        numer -= 2.0 * np.asarray(V0)
        return np.divide(numer,denom)
    