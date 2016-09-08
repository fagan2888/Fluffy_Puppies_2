# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 11:42:57 2016

@author: Chandler
"""

from __future__ import division
import pandas as pd
import numpy as np
import time


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

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap

# Calculate the CF table
#@timing
def pool_CF(principal, monthly_interest, payment_number, PSA, maturity, age, SMM = None):
    # Generate a blank dataframe for putting the data
    pool = {'PMT' : pd.Series(np.zeros(maturity * 12), index=range(1,maturity*12+1)), 
    'Interest' : pd.Series(np.zeros(maturity * 12), index=range(1,maturity*12+1)),
    'Principal' : pd.Series(np.zeros(maturity * 12), index=range(1,maturity*12+1)),
    'CPR' : pd.Series(np.zeros(maturity * 12), index=range(1,maturity*12+1)),
    'SMM' : pd.Series(np.zeros(maturity * 12), index=range(1,maturity*12+1)),
    'Prepayment' : pd.Series(np.zeros(maturity * 12), index=range(1,maturity*12+1)),
    'Remaining_principal' : pd.Series(np.zeros(maturity * 12), index=range(1,maturity*12+1))}
    pool_CF_df = pd.DataFrame(pool)
    
    SMM_empty = False
    
    if SMM is None:
        SMM_empty = True
        
    remaining_principal = principal
    for i in range(1, maturity * 12 + 1):
        # Calculate PMT, considering the zero divide
        if remaining_principal == 0 :
            monthly_payment = 0
        else:
            monthly_payment = remaining_principal * (monthly_interest * (1 + monthly_interest) ** (payment_number - i + 1))/((1 + monthly_interest) ** (payment_number - i + 1) - 1)   
        # calculate interest payment
        pi = remaining_principal * monthly_interest    
        # calculate principal payment
        if (remaining_principal - (monthly_payment - pi))>0.001:
            pp = monthly_payment - pi
        else:
            pp = remaining_principal
        # calculate prepayment
        prepayment_CPR = PSA * 0.06 * min(1,(age+i)/30)
        if SMM_empty:
            SMM = 1-(1-prepayment_CPR)**(1/12)
            prepayment = SMM * (remaining_principal - pp)
        else:
            prepayment = SMM[i-1] * (remaining_principal - pp)
            
        # get the remaining principal
        remaining_principal = remaining_principal - pp - prepayment
        # Plug in the data
        pool_CF_df.loc[i,'PMT'] = monthly_payment
        pool_CF_df.loc[i,'Interest'] = pi
        pool_CF_df.loc[i,'Principal'] = pp
        pool_CF_df.loc[i,'CPR'] = prepayment_CPR
        
        if SMM_empty:
            pool_CF_df.loc[i,'SMM'] = SMM
        else:
            pool_CF_df.loc[i,'SMM'] = SMM[i-1]
            
        pool_CF_df.loc[i,'Prepayment'] = prepayment
        pool_CF_df.loc[i,'Remaining_principal'] = remaining_principal
        #print "%d\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f" % (i, monthly_payment, pi, pp, prepayment_CPR, SMM, prepayment, remaining_principal)
    return pool_CF_df


# Summary CF
#@timing
def summary_CF(pool_CF_1, pool_CF_2):
    # Generate a blank dataframe

    summary = {'Total_Principal' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                  'Total_Interest' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                  'EOM_Balance' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                  'Int_Available_CMO' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1))
                  }
    summary_df = pd.DataFrame(summary)
    for i in range(1, maturity_1 * 12 + 1):
        summary_df.loc[i,'Total_Principal'] = pool_CF_1.loc[i,'Principal'] + pool_CF_1.loc[i,'Prepayment'] + pool_CF_2.loc[i,'Principal'] + pool_CF_2.loc[i,'Prepayment']
        summary_df.loc[i,'Total_Interest'] = pool_CF_1.loc[i,'Interest'] + pool_CF_2.loc[i,'Interest']
        summary_df.loc[i,'EOM_Balance'] = pool_CF_1.loc[i,'Remaining_principal'] + pool_CF_2.loc[i,'Remaining_principal']
        if i == 1:
            summary_df.loc[i,'Int_Available_CMO'] = (principal_1 + principal_2) * 0.05 / 12
        else:
            summary_df.loc[i,'Int_Available_CMO'] = summary_df.loc[i-1,'EOM_Balance'] * 0.05 / 12
        #print "%d\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t" % (i,summary_CF_df.loc[i,'Total_Principal'],summary_CF_df.loc[i,'Total_Interest'],summary_CF_df.loc[i,'EOM_Balance'],summary_CF_df.loc[i,'Int_Available_CMO'])
    return summary_df

# Principal CF Alloc
#@timing
def Principal_CF_Alloc(summary_CF_df,tranche_coupon,CG,VE,CM,GZ,TC,CZ,CA,CY):
    proportion_1 = (CG+VE+CM+GZ+TC+CZ)/(CG+VE+CM+GZ+TC+CZ+CA+CY)
    proportion_2 = (CA+CY)/(CG+VE+CM+GZ+TC+CZ+CA+CY)
    # Generate blank data frame
    col_headers = ['Principal_1', 'CG_Principal', 'CG_EOM', 'VE_Principal', 'VE_EOM', 'CM_Principal', 'CM_EOM',
                   'GZ_Interest', 'GZ_Accrued', 'GZ_Principal', 'GZ_EOM', 'TC_Principal', 'TC_EOM', 'CZ_Interest',
                   'CZ_Accrued', 'CZ_Principal', 'CZ_EOM', 'Principal_2', 'CA_Principal', 'CA_EOM', 'CY_Principal','CY_EOM']

    principal_CF_Alloc_df = pd.DataFrame(np.zeros((maturity_1 * 12, len(col_headers))), columns=col_headers, index=range(1,maturity_1*12+1))
    principal_CF_Alloc_df['Principal_1'] = summary_CF_df['Total_Principal'] * proportion_1
    for i in range(1, maturity_1 * 12 + 1):
        if i == 1:
            principal_CF_Alloc_df.loc[i,'CZ_Interest'] = CZ * tranche_coupon
            principal_CF_Alloc_df.loc[i,'CG_Principal'] = min(principal_CF_Alloc_df.loc[i,'Principal_1'] + principal_CF_Alloc_df.loc[i,'CZ_Interest'],CG)
            principal_CF_Alloc_df.loc[i,'CG_EOM'] = CG - principal_CF_Alloc_df.loc[i,'CG_Principal']
            principal_CF_Alloc_df.loc[i,'GZ_Interest'] = GZ * tranche_coupon
            principal_CF_Alloc_df.loc[i,'VE_Principal'] = max(0,min(principal_CF_Alloc_df.loc[i,'Principal_1']+principal_CF_Alloc_df.loc[i,'GZ_Interest']+principal_CF_Alloc_df.loc[i,'CZ_Interest'] - principal_CF_Alloc_df.loc[i,'CG_Principal'],VE))
            principal_CF_Alloc_df.loc[i,'VE_EOM'] = VE - principal_CF_Alloc_df.loc[i,'VE_Principal']
            principal_CF_Alloc_df.loc[i,'CM_Principal'] = max(0,min(principal_CF_Alloc_df.loc[i,'Principal_1']+principal_CF_Alloc_df.loc[i,'GZ_Interest']+principal_CF_Alloc_df.loc[i,'CZ_Interest'] - principal_CF_Alloc_df.loc[i,'CG_Principal'] - principal_CF_Alloc_df.loc[i,'VE_Principal'],CM))
            principal_CF_Alloc_df.loc[i,'CM_EOM'] = max(0,CM - principal_CF_Alloc_df.loc[i,'CM_Principal'])
            if principal_CF_Alloc_df.loc[i,'CM_EOM'] >0:
                principal_CF_Alloc_df.loc[i,'GZ_Accrued'] = principal_CF_Alloc_df.loc[i,'GZ_Interest']
            else:
                principal_CF_Alloc_df.loc[i,'GZ_Accrued'] = min(principal_CF_Alloc_df.loc[i,'CM_Principal'],principal_CF_Alloc_df.loc[i,'GZ_Interest'])
            principal_CF_Alloc_df.loc[i,'GZ_Principal'] = max(0,min(principal_CF_Alloc_df.loc[i,'Principal_1']+principal_CF_Alloc_df.loc[i,'GZ_Accrued']+principal_CF_Alloc_df.loc[i,'CZ_Interest']-principal_CF_Alloc_df.loc[i,'VE_Principal']-principal_CF_Alloc_df.loc[i,'CG_Principal']-principal_CF_Alloc_df.loc[i,'CM_Principal'],GZ))
            principal_CF_Alloc_df.loc[i,'GZ_EOM'] = max(GZ + principal_CF_Alloc_df.loc[i,'GZ_Accrued']-principal_CF_Alloc_df.loc[i,'GZ_Principal'],0)
            if principal_CF_Alloc_df.loc[i,'GZ_EOM'] >0:
                principal_CF_Alloc_df.loc[i,'TC_Principal'] = 0
            else:
                principal_CF_Alloc_df.loc[i,'TC_Principal'] = min(principal_CF_Alloc_df.loc[i,'Principal_1'] + principal_CF_Alloc_df.loc[i,'CZ_Interest'] - principal_CF_Alloc_df.loc[i,'GZ_Principal'] ,TC)
            principal_CF_Alloc_df.loc[i,'TC_EOM'] = max(0, TC-principal_CF_Alloc_df.loc[i,'TC_Principal'])
            if principal_CF_Alloc_df.loc[i,'TC_EOM'] >0:
                principal_CF_Alloc_df.loc[i,'CZ_Accrued'] = principal_CF_Alloc_df.loc[i,'CZ_Interest']
            else:
                principal_CF_Alloc_df.loc[i,'CZ_Accrued'] = min(principal_CF_Alloc_df.loc[i,'TC_Principal'],principal_CF_Alloc_df.loc[i,'CZ_Interest'])
            principal_CF_Alloc_df.loc[i,'CZ_Principal'] = max(0,min(principal_CF_Alloc_df.loc[i,'Principal_1']+principal_CF_Alloc_df.loc[i,'CZ_Accrued']-principal_CF_Alloc_df.loc[i,'VE_Principal']-principal_CF_Alloc_df.loc[i,'CG_Principal']-principal_CF_Alloc_df.loc[i,'CM_Principal']-principal_CF_Alloc_df.loc[i,'GZ_Principal']-principal_CF_Alloc_df.loc[i,'TC_Principal'],CZ))
            principal_CF_Alloc_df.loc[i,'CZ_EOM'] = max(0,CZ + principal_CF_Alloc_df.loc[i,'CZ_Accrued'] - principal_CF_Alloc_df.loc[i,'CZ_Principal'])
            principal_CF_Alloc_df.loc[i,'Principal_2'] = summary_CF_df.loc[i,'Total_Principal'] * proportion_2         
            principal_CF_Alloc_df.loc[i,'CA_Principal'] = min(principal_CF_Alloc_df.loc[i,'Principal_2'],CA)
            principal_CF_Alloc_df.loc[i,'CA_EOM'] = CA - principal_CF_Alloc_df.loc[i,'CA_Principal']
            principal_CF_Alloc_df.loc[i,'CY_Principal'] = min(principal_CF_Alloc_df.loc[i,'Principal_2'] - principal_CF_Alloc_df.loc[i,'CA_Principal'],CY)
            principal_CF_Alloc_df.loc[i,'CY_EOM'] = CY - principal_CF_Alloc_df.loc[i,'CY_Principal']           
            
        else:
            principal_CF_Alloc_df.loc[i,'CZ_Interest'] = principal_CF_Alloc_df.loc[i-1,'CZ_EOM'] * tranche_coupon
            principal_CF_Alloc_df.loc[i,'CG_Principal'] = min(principal_CF_Alloc_df.loc[i,'Principal_1']+principal_CF_Alloc_df.loc[i,'CZ_Interest'],principal_CF_Alloc_df.loc[i-1,'CG_EOM'])
            principal_CF_Alloc_df.loc[i,'CG_EOM'] =  principal_CF_Alloc_df.loc[i-1,'CG_EOM']- principal_CF_Alloc_df.loc[i,'CG_Principal']
            principal_CF_Alloc_df.loc[i,'GZ_Interest'] = principal_CF_Alloc_df.loc[i-1,'GZ_EOM'] * tranche_coupon
            principal_CF_Alloc_df.loc[i,'VE_Principal'] = max(0,min(principal_CF_Alloc_df.loc[i,'Principal_1']+principal_CF_Alloc_df.loc[i,'GZ_Interest']+principal_CF_Alloc_df.loc[i,'CZ_Interest']-principal_CF_Alloc_df.loc[i,'CG_Principal'],principal_CF_Alloc_df.loc[i-1,'VE_EOM']))              
            principal_CF_Alloc_df.loc[i,'VE_EOM'] =  principal_CF_Alloc_df.loc[i-1,'VE_EOM']- principal_CF_Alloc_df.loc[i,'VE_Principal']
            principal_CF_Alloc_df.loc[i,'CM_Principal'] = max(0,min(principal_CF_Alloc_df.loc[i,'Principal_1']+principal_CF_Alloc_df.loc[i,'GZ_Interest']+principal_CF_Alloc_df.loc[i,'CZ_Interest']-principal_CF_Alloc_df.loc[i,'CG_Principal'] - principal_CF_Alloc_df.loc[i,'VE_Principal'], principal_CF_Alloc_df.loc[i-1,'CM_EOM']))
            principal_CF_Alloc_df.loc[i,'CM_EOM'] = max(0, principal_CF_Alloc_df.loc[i-1,'CM_EOM'] - principal_CF_Alloc_df.loc[i,'CM_Principal'])
            if principal_CF_Alloc_df.loc[i,'CM_EOM'] >0:
                principal_CF_Alloc_df.loc[i,'GZ_Accrued'] = principal_CF_Alloc_df.loc[i,'GZ_Interest']
            else:
                principal_CF_Alloc_df.loc[i,'GZ_Accrued'] = min(principal_CF_Alloc_df.loc[i,'CM_Principal'],principal_CF_Alloc_df.loc[i,'GZ_Interest'])
            principal_CF_Alloc_df.loc[i,'GZ_Principal'] = max(0,min(principal_CF_Alloc_df.loc[i,'Principal_1']+principal_CF_Alloc_df.loc[i,'GZ_Accrued']+principal_CF_Alloc_df.loc[i,'CZ_Interest']-principal_CF_Alloc_df.loc[i,'VE_Principal']-principal_CF_Alloc_df.loc[i,'CG_Principal']-principal_CF_Alloc_df.loc[i,'CM_Principal'],principal_CF_Alloc_df.loc[i-1,'GZ_EOM']))
            principal_CF_Alloc_df.loc[i,'GZ_EOM'] = max(principal_CF_Alloc_df.loc[i-1,'GZ_EOM'] + principal_CF_Alloc_df.loc[i,'GZ_Accrued']-principal_CF_Alloc_df.loc[i,'GZ_Principal'],0)
            if principal_CF_Alloc_df.loc[i,'GZ_EOM'] >0:
                principal_CF_Alloc_df.loc[i,'TC_Principal'] = 0
            else:
                principal_CF_Alloc_df.loc[i,'TC_Principal'] = min(principal_CF_Alloc_df.loc[i,'Principal_1']+principal_CF_Alloc_df.loc[i,'CZ_Interest']-principal_CF_Alloc_df.loc[i,'GZ_Principal'], principal_CF_Alloc_df.loc[i-1,'TC_EOM'])
            principal_CF_Alloc_df.loc[i,'TC_EOM'] = max(0, principal_CF_Alloc_df.loc[i-1,'TC_EOM']-principal_CF_Alloc_df.loc[i,'TC_Principal'])
            if principal_CF_Alloc_df.loc[i,'TC_EOM'] >0:
                principal_CF_Alloc_df.loc[i,'CZ_Accrued'] = principal_CF_Alloc_df.loc[i,'CZ_Interest']
            else:
                principal_CF_Alloc_df.loc[i,'CZ_Accrued'] = min(principal_CF_Alloc_df.loc[i,'TC_Principal'],principal_CF_Alloc_df.loc[i,'CZ_Interest'])
            principal_CF_Alloc_df.loc[i,'CZ_Principal'] = max(0,min(principal_CF_Alloc_df.loc[i,'Principal_1']+principal_CF_Alloc_df.loc[i,'CZ_Accrued']-principal_CF_Alloc_df.loc[i,'VE_Principal']-principal_CF_Alloc_df.loc[i,'CG_Principal']-principal_CF_Alloc_df.loc[i,'CM_Principal']-principal_CF_Alloc_df.loc[i,'GZ_Principal']-principal_CF_Alloc_df.loc[i,'TC_Principal'],principal_CF_Alloc_df.loc[i-1,'CZ_EOM']))
            principal_CF_Alloc_df.loc[i,'CZ_EOM'] = max(0, principal_CF_Alloc_df.loc[i-1,'CZ_EOM']+ principal_CF_Alloc_df.loc[i,'CZ_Accrued'] - principal_CF_Alloc_df.loc[i,'CZ_Principal'])
            principal_CF_Alloc_df.loc[i,'Principal_2'] = summary_CF_df.loc[i,'Total_Principal'] * proportion_2              
            principal_CF_Alloc_df.loc[i,'CA_Principal'] = min(principal_CF_Alloc_df.loc[i,'Principal_2'],principal_CF_Alloc_df.loc[i-1,'CA_EOM'])
            principal_CF_Alloc_df.loc[i,'CA_EOM'] = principal_CF_Alloc_df.loc[i-1,'CA_EOM'] - principal_CF_Alloc_df.loc[i,'CA_Principal']
            principal_CF_Alloc_df.loc[i,'CY_Principal'] = min(principal_CF_Alloc_df.loc[i,'Principal_2'] - principal_CF_Alloc_df.loc[i,'CA_Principal'],principal_CF_Alloc_df.loc[i-1,'CY_EOM'])
            principal_CF_Alloc_df.loc[i,'CY_EOM'] = principal_CF_Alloc_df.loc[i-1,'CY_EOM'] - principal_CF_Alloc_df.loc[i,'CY_Principal']
            
    return principal_CF_Alloc_df
            


    
#@timing
def Principal(principal_CF_Alloc):
    Principal = {'CG' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'VE' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'CM' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'GZ' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'TC' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'CZ' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'CA' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'CY' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'Total_Principal_less_Accrued_Interest' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1))}
    
    Principal_df = pd.DataFrame(Principal)
    for i in range(1, maturity_1 * 12 + 1):
        Principal_df.loc[i,'CG'] = principal_CF_Alloc.loc[i,'CG_Principal']
        Principal_df.loc[i,'VE'] = principal_CF_Alloc.loc[i,'VE_Principal']
        Principal_df.loc[i,'CM'] = principal_CF_Alloc.loc[i,'CM_Principal']
        Principal_df.loc[i,'GZ'] = principal_CF_Alloc.loc[i,'GZ_Principal']
        Principal_df.loc[i,'TC'] = principal_CF_Alloc.loc[i,'TC_Principal']
        Principal_df.loc[i,'CZ'] = principal_CF_Alloc.loc[i,'CZ_Principal']
        Principal_df.loc[i,'CA'] = principal_CF_Alloc.loc[i,'CA_Principal']
        Principal_df.loc[i,'CY'] = principal_CF_Alloc.loc[i,'CY_Principal']
        Principal_df.loc[i,'Total_Principal_less_Accrued_Interest'] = Principal_df.loc[i,'CG']+Principal_df.loc[i,'VE']+Principal_df.loc[i,'CM']+Principal_df.loc[i,'GZ']+Principal_df.loc[i,'TC']+Principal_df.loc[i,'CZ']+Principal_df.loc[i,'CA']+Principal_df.loc[i,'CY']-principal_CF_Alloc.loc[i,'GZ_Accrued']-principal_CF_Alloc.loc[i,'CZ_Accrued']
        #print "%d\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t" % (i,Principal_df.loc[i,'CG'],Principal_df.loc[i,'VE'],Principal_df.loc[i,'CM'],Principal_df.loc[i,'GZ'],Principal_df.loc[i,'TC'],Principal_df.loc[i,'CZ'],Principal_df.loc[i,'CA'],Principal_df.loc[i,'CY'],Principal_df.loc[i,'Total_Principal_less_Accrued_Interest'])
    return Principal_df
   
#@timing
def Balance(summary_CF,principal_CF_Alloc):
    Balance = {'CG' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'VE' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'CM' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'GZ' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'TC' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'CZ' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'CA' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'CY' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'Total' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1))}
    Balance_df = pd.DataFrame(Balance)
    Balance_df.loc[1,'CG'] = CG
    Balance_df.loc[1,'VE'] = VE
    Balance_df.loc[1,'CM'] = CM
    Balance_df.loc[1,'GZ'] = GZ
    Balance_df.loc[1,'TC'] = TC
    Balance_df.loc[1,'CZ'] = CZ
    Balance_df.loc[1,'CA'] = CA
    Balance_df.loc[1,'CY'] = CY
    Balance_df.loc[1,'Total'] = Balance_df.loc[1,'CG']+Balance_df.loc[1,'VE']+Balance_df.loc[1,'CM']+Balance_df.loc[1,'GZ']+Balance_df.loc[1,'TC']+Balance_df.loc[1,'CZ']+Balance_df.loc[1,'CA']+Balance_df.loc[1,'CY']

    for i in range(1, maturity_1 * 12 + 1):
        Balance_df.loc[i+1,'CG'] = principal_CF_Alloc.loc[i,'CG_EOM']
        Balance_df.loc[i+1,'VE'] = principal_CF_Alloc.loc[i,'VE_EOM']
        Balance_df.loc[i+1,'CM'] = principal_CF_Alloc.loc[i,'CM_EOM']
        Balance_df.loc[i+1,'GZ'] = principal_CF_Alloc.loc[i,'GZ_EOM']
        Balance_df.loc[i+1,'TC'] = principal_CF_Alloc.loc[i,'TC_EOM']
        Balance_df.loc[i+1,'CZ'] = principal_CF_Alloc.loc[i,'CZ_EOM']
        Balance_df.loc[i+1,'CA'] = principal_CF_Alloc.loc[i,'CA_EOM']
        Balance_df.loc[i+1,'CY'] = principal_CF_Alloc.loc[i,'CY_EOM']
        Balance_df.loc[i+1,'Total'] = Balance_df.loc[i+1,'CG']+Balance_df.loc[i+1,'VE']+Balance_df.loc[i+1,'CM']+Balance_df.loc[i+1,'GZ']+Balance_df.loc[i+1,'TC']+Balance_df.loc[i+1,'CZ']+Balance_df.loc[i+1,'CA']+Balance_df.loc[i+1,'CY']
        #print "%d\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t" % (i,Balance_df.loc[i,'CG'],Balance_df.loc[i,'VE'],Balance_df.loc[i,'CM'],Balance_df.loc[i,'GZ'],Balance_df.loc[i,'TC'],Balance_df.loc[i,'CZ'],Balance_df.loc[i,'CA'],Balance_df.loc[i,'CY'],Balance_df.loc[i,'Total'])
    return Balance_df

#@timing
def Interest(balance,principal_CF_Alloc,summary_CF,tranche_coupon):
    Interest = {'CG' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'VE' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'CM' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'GZ' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'TC' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'CZ' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'CA' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'CY' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'Total' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'Checksum' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1))}
    Interest_df = pd.DataFrame(Interest)
    for i in range(1, maturity_1 * 12 + 1):
        Interest_df.loc[i,'CG'] = balance.loc[i,'CG'] * tranche_coupon
        Interest_df.loc[i,'VE'] = balance.loc[i,'VE'] * tranche_coupon
        Interest_df.loc[i,'CM'] = balance.loc[i,'CM'] * tranche_coupon
        Interest_df.loc[i,'GZ'] = principal_CF_Alloc.loc[i,'GZ_Interest'] - principal_CF_Alloc.loc[i,'GZ_Accrued']
        Interest_df.loc[i,'TC'] = balance.loc[i,'TC'] * tranche_coupon
        Interest_df.loc[i,'CZ'] = principal_CF_Alloc.loc[i,'CZ_Interest'] - principal_CF_Alloc.loc[i,'CZ_Accrued']
        Interest_df.loc[i,'CA'] = balance.loc[i,'CA'] * tranche_coupon
        Interest_df.loc[i,'CY'] = balance.loc[i,'CY'] * tranche_coupon
        Interest_df.loc[i,'Total'] = Interest_df.loc[i,'CG']+Interest_df.loc[i,'VE']+Interest_df.loc[i,'CM']+Interest_df.loc[i,'GZ']+Interest_df.loc[i,'TC']+Interest_df.loc[i,'CZ']+Interest_df.loc[i,'CA']+Interest_df.loc[i,'CY']
        #Interest_df.loc[i,'Checksum'] = summary_CF[i,'Int_Available_CMO'] - Interest_df.loc[i,'Total'] 
        #print "%d\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t" % (i,Interest_df.loc[i,'CG'],Interest_df.loc[i,'VE'],Interest_df.loc[i,'CM'],Interest_df.loc[i,'GZ'],Interest_df.loc[i,'TC'],Interest_df.loc[i,'CZ'],Interest_df.loc[i,'CA'],Interest_df.loc[i,'CY'],Interest_df.loc[i,'Total'])
    return Interest_df    

#@timing
def Pricing(principal,interest):
    # need to adjust index temporately for adjusting the HW1.py calculation

    Pricing = {'Period' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'CG' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'VE' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'CM' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'GZ' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                         'TC' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'CZ' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'CA' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1)),
                          'CY' : pd.Series(np.zeros(maturity_1 * 12), index=range(1,maturity_1*12+1))}


    Pricing_df = pd.DataFrame(Pricing)    
    for i in range(1, maturity_1 * 12+1):
        Pricing_df.loc[i,'Period'] = i
        Pricing_df.loc[i,'CG'] = principal.loc[i,'CG'] + interest.loc[i,'CG']
        Pricing_df.loc[i,'VE'] = principal.loc[i,'VE'] + interest.loc[i,'VE']
        Pricing_df.loc[i,'CM'] = principal.loc[i,'CM'] + interest.loc[i,'CM']
        Pricing_df.loc[i,'GZ'] = principal.loc[i,'GZ'] + interest.loc[i,'GZ']
        Pricing_df.loc[i,'TC'] = principal.loc[i,'TC'] + interest.loc[i,'TC']
        Pricing_df.loc[i,'CZ'] = principal.loc[i,'CZ'] + interest.loc[i,'CZ']
        Pricing_df.loc[i,'CA'] = principal.loc[i,'CA'] + interest.loc[i,'CA']
        Pricing_df.loc[i,'CY'] = principal.loc[i,'CY'] + interest.loc[i,'CY']
        #print("%d\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t" % (i,Pricing_df.loc[i,'Period'],Pricing_df.loc[i,'CG'],Pricing_df.loc[i,'VE'],Pricing_df.loc[i,'CM'],Pricing_df.loc[i,'GZ'],Pricing_df.loc[i,'TC'],Pricing_df.loc[i,'CZ'],Pricing_df.loc[i,'CA'],Pricing_df.loc[i,'CY']))
    return Pricing_df
    
    
    
pool_CF_1 = pool_CF(principal_1, monthly_interest_1, payment_number_1, PSA_1, maturity_1, age_1)
pool_CF_2 = pool_CF(principal_2, monthly_interest_2, payment_number_2, PSA_2, maturity_2, age_2)
summary = summary_CF(pool_CF_1, pool_CF_2)
principal_CF_Alloc = Principal_CF_Alloc(summary,tranche_coupon,CG,VE,CM,GZ,TC,CZ,CA,CY)
principal = Principal(principal_CF_Alloc)
balance = Balance(summary,principal_CF_Alloc)
interest = Interest(balance,principal_CF_Alloc,summary,tranche_coupon)
pricing = Pricing(principal,interest)



cols = pricing.columns.tolist()
cols = cols[6:7] + cols[1:2] + cols[8:9] + cols[2:3] + cols[5:6] + cols[7:8] + cols[4:5] + cols[0:1] + cols[3:4]
data_cashflow = pricing[cols]