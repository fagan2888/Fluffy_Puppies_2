#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 18:53:42 2016

@author: kunmingwu
"""

import pandas as pd
import numpy as np
import scipy as sp
import Hull_White as hw
import mortgage_np as m
import Discount_Functions as disc_func
import math

### Variables Setup

cap_rate = 0.0475      # given in HW2_Data file
short_rate = 0.01816   # given in HW2_Data file
dt = 0.25
r0 = short_rate

HW = hw.Hull_White()   #our Hull White class
K = 1/(1+cap_rate*dt)  #strike in put model

# from HW1 problem set
kappa = 0.153
sigma = 0.0153

# from HW1 REMIC data file
WACs = [0.05402, 0.05419]   # WAC values for two pools

df_stmat = pd.read_csv('stmat.csv')
df_tvmat = pd.read_csv('tvmat.csv')
df_disc_factors = pd.read_csv('discount factors.csv')
df_caplet_vols = pd.read_csv('caplet vols.csv')
df_mthly_10_yr_LIBOR_04 = pd.read_csv('monthly 10 year LIBOR spot rates for 2004.csv')

df_stmat['Coupon Gap'] = df_stmat['Coupon Gap'] * 0.0001 # convert from basis points to decimal
df_tvmat['Coupon Gap'] = df_tvmat['Coupon Gap'] * 0.0001 # convert from basis points to decimal

## import and clean data
data_z = pd.read_csv("discount factors.csv", header = 0)

## fit Z curve using OLS
data_z["poly"] = np.log(data_z["Price"])
z_OLS = disc_func.OLS(disc_func.power_5(data_z["Maturity"]), data_z["poly"])

(a,b,c,d,e) = z_OLS.beta
coeff = [a,b,c,d,e]


gamma = 0.0568
p = 2.9025
beta_1 = -12.6051
beta_2 = -2.0992

gamma_se = 0.0018
p_se = 0.0750
beta_1_se = 1.9998
beta_2_se = 0.0495

beta = np.array([beta_1, beta_2])

# 10 simulation, so that the total number of paths is 100
# FOR HIGHER PRECISION, CHANGE TO A LARGE NUMBER
num_sims = 1000
T = 20
dt = 1.0/12.0
dr = 10e-5  # dr taken to be arbitrarily small


# Excel Sheet Setup
# basic information of two mortgage, for mortgage package
principal_1 = 77657656.75
monthly_interest_1 = 5.402/1200
payment_number_1 = 236
PSA_1 = 1.5
maturity_1 = 20
age_1 = 3

principal_2 = 128842343.35
monthly_interest_2 = 5.419/1200
payment_number_2 = 237
PSA_2 = 1.5
maturity_2 = 20
age_2 = 3

CG = 74800000
VE = 5200000
CM = 14000000
GZ = 22000000
TC = 20000000
CZ = 24000000
CA = 32550000
CY = 13950000
tranche_coupon = 5/(12*100)
par_price = np.array((CG,VE,CM,GZ,TC,CZ,CA,CY))


def libor_rate_10yr_lag3_from_MC(r_arr):
    libor_arr = [0.0539, 0.0522, 0.0510]
    t_range = np.arange(1,361-3-12*10)
    for t in t_range:
        r_1 = r_arr[t]
        r_10 = r_arr[t+12*10]
        libor_arr.append((r_10*(t+10*12) - r_1*(t)) / (t+10*12 - t))
    return libor_arr

def summer_index_func(t):
    # starting from Sept. plus 8
    t = t + 8
    return 1 if t%12 in [5,6,7,8] else 0

def hazard_func_arr(gamma, p, beta, pool, libor_arr):
    #here the time input t should be integer month
    t_range = np.arange(1,241)
    libor_arr = np.asarray(libor_arr[0]).flatten()
    summer_index_arr = [summer_index_func(t) for t in t_range]
    v = np.array([WACs[pool] - libor_arr, summer_index_arr])
    v = np.transpose(v)
    exp_val = np.dot(v, beta)
    up = (gamma * p * (gamma * t_range)**(p-1))
    down = (1 + (gamma * t_range)**p)
    mid = np.exp(exp_val) 
    #return ((gamma*p * (gamma * t_range)**(p-1)) / (1 + (gamma * t_range)**p)) * np.exp(exp_val) 
    return np.multiply(np.divide(up,down),mid)
    
def SMM_func_arr(gamma, p, beta, pool, libor_arr):
    return (1-np.exp(-hazard_func_arr(gamma, p, beta, pool, libor_arr) * 1))

def get_libor_rate_matrix(r_matrix, num_sims):
    libor_rate_matrix = []
    for i in range(num_sims):
        libor_rate_matrix.append(libor_rate_10yr_lag3_from_MC(r_matrix[i,:]))
    libor_rate_matrix = np.matrix(libor_rate_matrix)
    return libor_rate_matrix

def get_SMM_matrices(gamma, p, beta, libor_rate_matrix, num_sims):
    SMM_matrix_1 = []
    SMM_matrix_2 = []

    for i in range(num_sims):
        libor_arr = libor_rate_matrix[i,:]
        SMM_matrix_1.append(pd.Series(SMM_func_arr(gamma, p, beta, 0, libor_arr)))
        SMM_matrix_2.append(pd.Series(SMM_func_arr(gamma, p, beta, 1, libor_arr)))

    return (SMM_matrix_1, SMM_matrix_2)

def get_bond_prices_matrix(num_sims, SMM_matrix_1, SMM_matrix_2, cum_df_matrix):
    pricing_arr = []
    for i in range(num_sims):
        if i % int(num_sims//100 + 1) ==0:
            print('.',end='')
        SMM_arr_1 = SMM_matrix_1[i]
        SMM_arr_2 = SMM_matrix_2[i]
        pool_CF_1 = m.pool_CF(principal_1, monthly_interest_1, payment_number_1, PSA_1, maturity_1, age_1, SMM_arr_1)
        pool_CF_2 = m.pool_CF(principal_2, monthly_interest_2, payment_number_2, PSA_2, maturity_2, age_2, SMM_arr_2)
        summary_CF = m.summary_CF(pool_CF_1, pool_CF_2)
        principal_CF_Alloc = m.Principal_CF_Alloc(summary_CF,tranche_coupon,CG,VE,CM,GZ,TC,CZ,CA,CY)
        principal = m.Principal(principal_CF_Alloc)
        balance = m.Balance(summary_CF,principal_CF_Alloc)
        interest = m.Interest(balance,principal_CF_Alloc,summary_CF,tranche_coupon)
        pricing = m.Pricing(principal,interest)
        pricing_arr.append(pricing)

    bond_prices_matrix = []
    for i in range(num_sims):
        pricing = pricing_arr[i]
        # Adjsut the order of column 
        cols = (pricing_arr[0]).columns
        cols = cols[:len(cols)-1]
        data_cashflow = pricing[cols]

        # Price bonds
        bonds = data_cashflow
        #bond_prices = 0.5 * ((cum_df_matrix[i] * np.matrix(bonds)) + (cum_df_anti_matrix[i] * np.matrix(bonds)))
        bond_prices = cum_df_matrix[i] * np.matrix(bonds)
        bond_prices = np.asarray(bond_prices)[0]
        bond_prices = [float('%.3f' % x) for x in bond_prices]


        bond_prices_matrix.append(bond_prices)
    bond_prices_matrix = np.matrix(bond_prices_matrix)
    print('')
    return (bond_prices_matrix, summary_CF, pricing_arr)

def get_V(r0):
    ## Simlation:
    ## cum_df_matrix = cumulative discount factor matrix
    ## cum_df_anti_matrix = cumulative discount factor matrix with antithetic path

    ## T=30 since we need to calculate 10yr libor rate (20+10=30)
    (cum_df_matrix, cum_df_anti_matrix, r_matrix, r_anti_matrix) = HW.Monte_Carlo_2(kappa, sigma, r0, 30, dt, coeff, num_sims)
    r_matrix_original = r_matrix
    r_anti_matrix_original = r_anti_matrix
    # we only need 240 period discount factor, using 360 to get short rate -> to get libor later
    cum_df_matrix = cum_df_matrix[:,:240]
    cum_df_anti_matrix = cum_df_anti_matrix[:,:240]

    libor_rate_matrix = get_libor_rate_matrix(r_matrix, num_sims)
    libor_rate_matrix_anti = get_libor_rate_matrix(r_anti_matrix, num_sims)
    #After we are done with LIBOR, get rid of the remaining r_matrix
    r_matrix = r_matrix[:,:240]
    r_anti_matrix = r_anti_matrix[:,:240]

    (SMM_matrix_1, SMM_matrix_2) = get_SMM_matrices(gamma, p, beta, libor_rate_matrix, num_sims)
    (SMM_matrix_1_anti, SMM_matrix_2_anti) = get_SMM_matrices(gamma, p, beta, libor_rate_matrix_anti, num_sims)


    (bond_prices_matrix_anti, summary_CF_anti, pricing_arr_anti) = get_bond_prices_matrix(num_sims, SMM_matrix_1_anti, SMM_matrix_2_anti, cum_df_matrix)
    (bond_prices_matrix, summary_CF, pricing_arr) = get_bond_prices_matrix(num_sims, SMM_matrix_1, SMM_matrix_2, cum_df_anti_matrix)

    final_bond_prices_matrix = 0.5 * (bond_prices_matrix_anti + bond_prices_matrix)
    bond_prices = np.asarray(final_bond_prices_matrix.mean(0))[0]
    standard_errors = np.asarray(final_bond_prices_matrix.std(0))[0] / math.sqrt(num_sims)

    return (bond_prices, standard_errors, summary_CF, summary_CF_anti, 
            r_matrix, r_anti_matrix, cum_df_matrix, cum_df_anti_matrix,
            pricing_arr, pricing_arr_anti, r_matrix_original, r_anti_matrix_original)

def get_residual_V(r0,summary_CF, summary_CF_anti, r_matrix, r_anti_matrix, cum_df_matrix, cum_df_anti_matrix):

    resid_tranche = summary_CF['Total_Interest'] + summary_CF['Total_Principal']
    resid_tranche_anti = summary_CF_anti['Total_Interest'] + summary_CF_anti['Total_Principal']
    residual_cashflow = []
    residual_cashflow_anti = []
    for r in r_matrix:
        residual_cashflow.append(r * resid_tranche/24)
    for r in r_anti_matrix:
        residual_cashflow_anti.append(r * resid_tranche_anti/24)
    residual_cashflow = np.matrix(residual_cashflow)

    residual_arr = []
    residual_arr_anti = []
    for d in cum_df_matrix:
        residual_arr.append(np.dot(residual_cashflow,np.matrix(d).T))
    for d in cum_df_anti_matrix:
        residual_arr_anti.append(np.dot(residual_cashflow_anti,np.matrix(d).T))
    return 0.5 * (np.array(residual_arr) + np.array(residual_arr_anti))

def oas_obj_func(oas,bond,i,r_matrix):
    dt = 1.0/12.0
    iterations = 240
    r_matrix_new = r_matrix + oas
    df_matrix_new = np.exp(-r_matrix_new*dt)
    cum_df_matrix_new = np.zeros((num_sims, iterations))
    for j in range(num_sims):
        cum_df_matrix_new[j,:] = df_matrix_new[j,:].cumprod()
    return (((cum_df_matrix_new * (np.matrix(bond).T)).mean() - par_price[i])/1e9)**2 

def get_OAS(pricing_arr, pricing_arr_anti, r_matrix, r_matrix_anti):
    cols = (pricing_arr[0]).columns
    cols = cols[:len(cols)-1]
    #first 100 paths should be good enought to get a rough estimate
    oas_matrix = []
    for j in range(len(pricing_arr)):
        print('.',end='')
        pricing = pricing_arr[j]
        data_cashflow = pricing[cols]
        oas_arr = []
        for i in range(8):
            oas_res = sp.optimize.minimize(lambda oas: oas_obj_func(oas, data_cashflow.iloc[:,i],i,r_matrix),0.01)
            oas_arr.append(oas_res.x[0])
        oas_matrix.append(oas_arr)
        if j>=100:
            break
    print('')
    oas_matrix_anti = []     
    for j in range(len(pricing_arr_anti)):
        print('.',end='')
        pricing_anti = pricing_arr_anti[j]
        data_cashflow = pricing_anti[cols]
        oas_arr = []
        for i in range(8):
            oas_res = sp.optimize.minimize(lambda oas: oas_obj_func(oas, data_cashflow.iloc[:,i],i,r_matrix_anti),0.01)
            oas_arr.append(oas_res.x[0])
        oas_matrix_anti.append(oas_arr)
        if j>=100:
            break
    print('')
    final_oas_arr = np.asarray((0.5 * (np.matrix(oas_matrix) + np.matrix(oas_matrix_anti))).mean(0))[0]
    return final_oas_arr

def avg_hazard_rate(pool, r_matrix, r_anti_matrix, num_sims):
    hazard_rate_matrix = []
    libor_rate_matrix = get_libor_rate_matrix(r_matrix, num_sims)
    for i in range(np.shape(libor_rate_matrix)[0]):
        hazard_rate_matrix.append(hazard_func_arr(gamma, p, beta, pool, libor_rate_matrix[i,:]))
    hazard_rate_matrix_anti = []
    libor_rate_matrix_anti = get_libor_rate_matrix(r_anti_matrix, num_sims)
    for i in range(np.shape(libor_rate_matrix_anti)[0]):
        hazard_rate_matrix_anti.append(hazard_func_arr(gamma, p, beta, pool, libor_rate_matrix_anti[i,:]))

    final_hazard_rate_matrix = (np.matrix(hazard_rate_matrix) + np.matrix(hazard_rate_matrix_anti))*0.5
    #final_hazard_rate_matrix = np.matrix(hazard_rate_matrix)
    return np.asarray(final_hazard_rate_matrix.mean(0))

