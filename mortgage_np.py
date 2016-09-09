# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 16:01:52 2016

@author: linshanli
"""
from __future__ import absolute_import, division, print_function
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
def pool_CF(principal, monthly_interest, payment_number, PSA, maturity, age, SMM_ = None):
    headers = ['PMT', 'Interest', 'Principal', 'CPR', 'SMM', 'Prepayment', 'Remaining_principal']
    PMT = np.zeros(maturity * 12)
    Interest = np.zeros(maturity * 12)
    Principal = np.zeros(maturity * 12)
    CPR = np.zeros(maturity * 12)
    SMM = np.zeros(maturity * 12)
    Prepayment = np.zeros(maturity * 12)
    Remaining_principal = np.zeros(maturity * 12)
    
    SMM_empty = False
        
    if SMM_ is None:
        SMM_empty = True
        
    remaining_principal = principal
    for i in range(maturity * 12 ):
        # Calculate PMT, considering the zero divide
        if remaining_principal == 0 :
            monthly_payment = 0
        else:
            monthly_payment = remaining_principal * (monthly_interest * (1 + monthly_interest) ** (payment_number - i))/((1 + monthly_interest) ** (payment_number - i) - 1)   
        # calculate interest payment
        pi = remaining_principal * monthly_interest    
        # calculate principal payment
        if (remaining_principal - (monthly_payment - pi))>0.001:
            pp = monthly_payment - pi
        else:
            pp = remaining_principal
        # calculate prepayment
        prepayment_CPR = PSA * 0.06 * min(1,(age+i+1)/30)
        if SMM_empty:
            SMM_ = 1-(1-prepayment_CPR)**(1/12)
            prepayment = SMM_ * (remaining_principal - pp)
        else:
            prepayment = SMM_[i] * (remaining_principal - pp)
            
        # get the remaining principal
        remaining_principal = remaining_principal - pp - prepayment
        # Plug in the data
        PMT[i] = monthly_payment
        Interest[i] = pi
        Principal[i] = pp
        CPR[i] = prepayment_CPR
        
        if SMM_empty:
            SMM[i] = SMM_
        else:
            SMM[i] = SMM_[i]
            
        Prepayment[i] = prepayment
        Remaining_principal[i] = remaining_principal
    temp = np.column_stack((PMT,Interest,Principal,CPR,SMM,Prepayment,Remaining_principal))
    pool_CF_df = pd.DataFrame(temp, columns =headers)  
    return pool_CF_df
    

def summary_CF(pool_CF_1, pool_CF_2):
    # Generate numpy array
    headers = ['Total_Principal','Total_Interest','EOM_Balance','Int_Available_CMO']
    Total_Principal = np.zeros(maturity_1 * 12)
    Total_Interest = np.zeros(maturity_1 * 12)
    EOM_Balance = np.zeros(maturity_1 * 12)
    Int_Available_CMO = np.zeros(maturity_1 * 12)
    
    # Input the dataframe output from pool_CF and transfer it into the array, which is faster
    Principal_1 = np.array(pool_CF_1.loc[:,'Principal'])
    Prepayment_1 = np.array(pool_CF_1.loc[:,'Prepayment'])
    Interest_1 = np.array(pool_CF_1.loc[:,'Interest'])
    Remaining_principal_1 = np.array(pool_CF_1.loc[:,'Remaining_principal'])
    
    Principal_2 = np.array(pool_CF_2.loc[:,'Principal'])
    Prepayment_2 = np.array(pool_CF_2.loc[:,'Prepayment'])
    Interest_2 = np.array(pool_CF_2.loc[:,'Interest'])
    Remaining_principal_2 = np.array(pool_CF_2.loc[:,'Remaining_principal'])
    
    # Formula is base on excel sheet
    for i in range(0, maturity_1 * 12):
        Total_Principal[i] = Principal_1[i] + Prepayment_1[i] + Principal_2[i] + Prepayment_2[i]
        Total_Interest[i] = Interest_1[i] + Interest_2[i]
        EOM_Balance[i] = Remaining_principal_1[i] + Remaining_principal_2[i]
        if i == 0:
            Int_Available_CMO[i] = (principal_1 + principal_2) * 0.05 / 12
        else:
            Int_Available_CMO[i] = EOM_Balance[i-1] * 0.05 / 12
    temp = np.column_stack((Total_Principal,Total_Interest,EOM_Balance,Int_Available_CMO))
    pool_CF_df = pd.DataFrame(temp, columns =headers)     
    return pool_CF_df

   
def Principal_CF_Alloc(summary_CF_df,tranche_coupon,CG,VE,CM,GZ,TC,CZ,CA,CY):
    proportion_1 = (CG+VE+CM+GZ+TC+CZ)/(CG+VE+CM+GZ+TC+CZ+CA+CY)
    proportion_2 = (CA+CY)/(CG+VE+CM+GZ+TC+CZ+CA+CY)
    # Generate blank data frame
    col_headers = ['Principal_1', 'CG_Principal', 'CG_EOM', 'VE_Principal', 'VE_EOM', 'CM_Principal', 'CM_EOM',
                   'GZ_Interest', 'GZ_Accrued', 'GZ_Principal', 'GZ_EOM', 'TC_Principal', 'TC_EOM', 'CZ_Interest',
                   'CZ_Accrued', 'CZ_Principal', 'CZ_EOM', 'Principal_2', 'CA_Principal', 'CA_EOM', 'CY_Principal','CY_EOM']
    
    #principal_CF_Alloc_df = pd.DataFrame(np.zeros((maturity_1 * 12, len(col_headers))), columns=col_headers, index=range(1,maturity_1*12+1))
    #principal_CF_Alloc_df['Principal_1'] = summary_CF_df['Total_Principal'] * proportion_1
    Summary_CF_Total_Principal = np.array(summary_CF_df['Total_Principal'])
    m = len(Summary_CF_Total_Principal)
    Principal_1 = np.array(summary_CF_df['Total_Principal'] * proportion_1)
    CG_Principal = np.zeros(m)
    CG_EOM = np.zeros(m)
    VE_Principal = np.zeros(m)
    VE_EOM = np.zeros(m)
    CM_Principal = np.zeros(m)
    CM_EOM = np.zeros(m)
    GZ_Interest = np.zeros(m)
    GZ_Accrued = np.zeros(m)
    GZ_Principal = np.zeros(m)
    GZ_EOM = np.zeros(m)
    TC_Principal = np.zeros(m)
    TC_EOM = np.zeros(m)
    CZ_Interest = np.zeros(m)
    CZ_Accrued = np.zeros(m)
    CZ_Principal = np.zeros(m)
    CZ_EOM = np.zeros(m)
    Principal_2 = np.zeros(m)
    CA_EOM = np.zeros(m)
    CA_Principal = np.zeros(m)
    CA_EOM = np.zeros(m)
    CY_Principal = np.zeros(m)
    CY_EOM = np.zeros(m)
    
    
    
    for i in range(m):
        if i == 0:
            CZ_Interest[i] = CZ * tranche_coupon
            CG_Principal[i]= min(Principal_1[i]+CZ_Interest[i],CG)
            CG_EOM[i] = CG - CG_Principal[i]
            GZ_Interest[i] = GZ * tranche_coupon
            VE_Principal[i] = max(0,min(Principal_1[i]+GZ_Interest[i]+CZ_Interest [i] - CG_Principal[i],VE))
            VE_EOM[i] = VE - VE_Principal[i]
            CM_Principal[i] = max(0,min(Principal_1[i]+GZ_Interest[i]+CZ_Interest [i] - CG_Principal[i]- VE_Principal[i],CM))
            CM_EOM[i] = max(0,CM - CM_Principal[i])
            if CM_EOM[i] >0:
                GZ_Accrued[i] = GZ_Interest[i]
            else:
                GZ_Accrued[i] = min(CM_Principal[i],GZ_Interest[i])
            GZ_Principal[i] = max(0,min(Principal_1[i]+GZ_Accrued[i]+CZ_Interest [i]-VE_Principal[i]-CG_Principal[i]-CM_Principal[i],GZ))
            GZ_EOM[i] = max(GZ + GZ_Accrued[i]-GZ_Principal[i],0)
            if GZ_EOM[i] >0:
                TC_Principal[i] = 0
            else:
                TC_Principal[i] = min(Principal_1[i]+CZ_Interest[i] - GZ_Principal[i] ,TC)
            TC_EOM[i] = max(0, TC-TC_Principal[i])
            if TC_EOM[i] >0:
                CZ_Accrued [i] =CZ_Interest[i]
            else:
                CZ_Accrued [i] = min(TC_Principal[i],CZ_Interest [i])
            CZ_Principal[i] = max(0,min(Principal_1[i]+CZ_Accrued [i]-VE_Principal[i]-CG_Principal[i]-CM_Principal[i]-GZ_Principal[i]-TC_Principal[i],CZ))
            CZ_EOM [i] = max(0,CZ + CZ_Accrued [i] - CZ_Principal[i])
    
            Principal_2[i] = Summary_CF_Total_Principal[i] * proportion_2         
            CA_Principal[i] = min(Principal_2[i],CA)
            CA_EOM[i] = CA - CA_Principal[i]
            CY_Principal[i] = min(Principal_2[i] - CA_Principal[i],CY)
            CY_EOM[i] = CY - CY_Principal[i]           
            
        else:
            CZ_Interest[i] = CZ_EOM[i-1] * tranche_coupon
            CG_Principal[i]= min(Principal_1[i]+CZ_Interest [i],CG_EOM[i-1])
            CG_EOM[i] =  CG_EOM[i-1]- CG_Principal[i]
            GZ_Interest[i] = GZ_EOM[i-1] * tranche_coupon
            VE_Principal[i] = max(0,min(Principal_1[i]+GZ_Interest[i]+CZ_Interest [i]-CG_Principal[i],VE_EOM[i-1]))              
            VE_EOM[i] =  VE_EOM[i-1]- VE_Principal[i]
            CM_Principal[i] = max(0,min(Principal_1[i]+GZ_Interest[i]+CZ_Interest [i]-CG_Principal[i]- VE_Principal[i], CM_EOM[i-1]))
            CM_EOM[i] = max(0, CM_EOM[i-1] - CM_Principal[i])
            if CM_EOM[i] >0:
                GZ_Accrued[i] = GZ_Interest[i]
            else:
                GZ_Accrued[i] = min(CM_Principal[i],GZ_Interest[i])
            GZ_Principal[i] = max(0,min(Principal_1[i]+GZ_Accrued[i]+CZ_Interest [i]-VE_Principal[i]-CG_Principal[i]-CM_Principal[i],GZ_EOM[i-1]))
            GZ_EOM[i] = max(GZ_EOM[i-1] + GZ_Accrued[i]-GZ_Principal[i],0)
            if GZ_EOM[i] >0:
                TC_Principal[i] = 0
            else:
                TC_Principal[i] = min(Principal_1[i]+CZ_Interest [i]-GZ_Principal[i], TC_EOM[i-1])
            TC_EOM[i] = max(0, TC_EOM[i-1]-TC_Principal[i])
            if TC_EOM[i] >0:
                CZ_Accrued [i] =CZ_Interest[i]
            else:
                CZ_Accrued [i] = min(TC_Principal[i],CZ_Interest [i])
            CZ_Principal[i] = max(0,min(Principal_1[i]+CZ_Accrued [i]-VE_Principal[i]-CG_Principal[i]-CM_Principal[i]-GZ_Principal[i]-TC_Principal[i],CZ_EOM[i-1]))
            CZ_EOM [i] = max(0, CZ_EOM[i-1]+ CZ_Accrued [i] - CZ_Principal[i])
            Principal_2[i] = Summary_CF_Total_Principal[i] * proportion_2              
            CA_Principal[i] = min(Principal_2[i],CA_EOM[i-1])
            CA_EOM[i] = CA_EOM[i-1] - CA_Principal[i]
            CY_Principal[i] = min(Principal_2[i] - CA_Principal[i],CY_EOM[i-1])
            CY_EOM[i] = CY_EOM[i-1] - CY_Principal[i]
    temp = np.column_stack((Principal_1, CG_Principal, CG_EOM, VE_Principal, VE_EOM, CM_Principal, CM_EOM,
                   GZ_Interest, GZ_Accrued, GZ_Principal, GZ_EOM, TC_Principal, TC_EOM, CZ_Interest,
                   CZ_Accrued, CZ_Principal, CZ_EOM, Principal_2, CA_Principal, CA_EOM, CY_Principal,CY_EOM))
    Principal_CF_Alloc_df = pd.DataFrame(temp, columns = col_headers)
    return Principal_CF_Alloc_df


def Principal(principal_CF_Alloc):
    headers = ['CG','VE','CM','GZ','TC','CZ','CA','CY','Total_Principal_less_Accrued_Interest']
    CG = np.zeros(maturity_1 * 12)
    VE = np.zeros(maturity_1 * 12)
    CM = np.zeros(maturity_1 * 12)
    GZ = np.zeros(maturity_1 * 12)
    TC = np.zeros(maturity_1 * 12)
    CZ = np.zeros(maturity_1 * 12)
    CA = np.zeros(maturity_1 * 12)
    CY = np.zeros(maturity_1 * 12)
    Total_Principal_less_Accrued_Interest = np.zeros(maturity_1 * 12)
    
    # Translate all dataframe structure into numpy array which is faster to read and iterate
    CG_Principal = np.array(principal_CF_Alloc.loc[:,'CG_Principal'])
    VE_Principal = np.array(principal_CF_Alloc.loc[:,'VE_Principal'])
    CM_Principal = np.array(principal_CF_Alloc.loc[:,'CM_Principal'])
    GZ_Principal = np.array(principal_CF_Alloc.loc[:,'GZ_Principal'])
    TC_Principal = np.array(principal_CF_Alloc.loc[:,'TC_Principal'])
    CZ_Principal = np.array(principal_CF_Alloc.loc[:,'CZ_Principal'])
    CA_Principal = np.array(principal_CF_Alloc.loc[:,'CA_Principal'])
    CY_Principal = np.array(principal_CF_Alloc.loc[:,'CY_Principal'])
    GZ_Accrued = principal_CF_Alloc.loc[:,'GZ_Accrued']
    CZ_Accrued = principal_CF_Alloc.loc[:,'CZ_Accrued']
    
    # Start the calculation
    for i in range(0, maturity_1 * 12):
        CG[i] = CG_Principal[i]
        VE[i] = VE_Principal[i]
        CM[i] = CM_Principal[i]
        GZ[i] = GZ_Principal[i]
        TC[i] = TC_Principal[i]
        CZ[i] = CZ_Principal[i]
        CA[i] = CA_Principal[i]
        CY[i] = CY_Principal[i]
        Total_Principal_less_Accrued_Interest[i] = CG_Principal[i]+VE_Principal[i]+CM_Principal[i]+GZ_Principal[i]+TC_Principal[i]+CZ_Principal[i]+CA_Principal[i]+CY_Principal[i]-GZ_Accrued[i]-CZ_Accrued[i]
        #print "%d\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t" % (i,Principal_df.loc[i,'CG'],Principal_df.loc[i,'VE'],Principal_df.loc[i,'CM'],Principal_df.loc[i,'GZ'],Principal_df.loc[i,'TC'],Principal_df.loc[i,'CZ'],Principal_df.loc[i,'CA'],Principal_df.loc[i,'CY'],Principal_df.loc[i,'Total_Principal_less_Accrued_Interest'])
    temp = np.column_stack((CG,VE,CM,GZ,TC,CZ,CA,CY,Total_Principal_less_Accrued_Interest))
    Principal_df = pd.DataFrame(temp, columns =headers)
    return Principal_df

def Balance(summary_CF_df,principal_CF_Alloc):
    
    headers = ['CG','VE','CM','GZ','TC','CZ','CA','CY','Total']
    Balance_CG = np.zeros(maturity_1 * 12)
    Balance_VE = np.zeros(maturity_1 * 12)
    Balance_CM = np.zeros(maturity_1 * 12)
    Balance_GZ = np.zeros(maturity_1 * 12)
    Balance_TC = np.zeros(maturity_1 * 12)
    Balance_CZ = np.zeros(maturity_1 * 12)
    Balance_CA = np.zeros(maturity_1 * 12)
    Balance_CY = np.zeros(maturity_1 * 12)
    Balance_Total = np.zeros(maturity_1 * 12)    

    Balance_CG[0] = CG
    Balance_VE[0] = VE
    Balance_CM[0] = CM
    Balance_GZ[0] = GZ
    Balance_TC[0] = TC
    Balance_CZ[0] = CZ
    Balance_CA[0] = CA
    Balance_CY[0] = CY
    Balance_Total[0] = CG+VE+CM+GZ+TC+CZ+CA+CY
    
    CG_EOM = np.array(principal_CF_Alloc.loc[:,'CG_EOM'])
    VE_EOM = np.array(principal_CF_Alloc.loc[:,'VE_EOM'])
    CM_EOM = np.array(principal_CF_Alloc.loc[:,'CM_EOM'])
    GZ_EOM = np.array(principal_CF_Alloc.loc[:,'GZ_EOM'])
    TC_EOM = np.array(principal_CF_Alloc.loc[:,'TC_EOM'])
    CZ_EOM = np.array(principal_CF_Alloc.loc[:,'CZ_EOM'])
    CA_EOM = np.array(principal_CF_Alloc.loc[:,'CA_EOM'])
    CY_EOM = np.array(principal_CF_Alloc.loc[:,'CY_EOM'])        
    
    # Caluclate according the formula from excel sheet
    for i in range(0, maturity_1 * 12 - 1):
        Balance_CG[i+1] = CG_EOM[i]
        Balance_VE[i+1] = VE_EOM[i]
        Balance_CM[i+1] = CM_EOM[i]
        Balance_GZ[i+1] = GZ_EOM[i]
        Balance_TC[i+1] = TC_EOM[i]
        Balance_CZ[i+1] = CZ_EOM[i]
        Balance_CA[i+1] = CA_EOM[i]
        Balance_CY[i+1] = CY_EOM[i]
        Balance_Total[i+1] = Balance_CG[i+1]+Balance_VE[i+1]+Balance_CM[i+1]+Balance_GZ[i+1]+Balance_TC[i+1]+Balance_CZ[i+1]+Balance_CA[i+1]+Balance_CY[i+1]
        #print "%d\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t" % (i,Balance_df.loc[i,'CG'],Balance_df.loc[i,'VE'],Balance_df.loc[i,'CM'],Balance_df.loc[i,'GZ'],Balance_df.loc[i,'TC'],Balance_df.loc[i,'CZ'],Balance_df.loc[i,'CA'],Balance_df.loc[i,'CY'],Balance_df.loc[i,'Total'])
    temp = np.column_stack((Balance_CG,Balance_VE,Balance_CM,Balance_GZ,Balance_TC,Balance_CZ,Balance_CA,Balance_CY,Balance_Total))
    Balance_df = pd.DataFrame(temp, columns =headers)    
    
    return Balance_df  

def Interest(balance,principal_CF_Alloc,summary_CF_df,tranche_coupon):
    # Generate the numpy array    
    headers = ['CG','VE','CM','GZ','TC','CZ','CA','CY','Total','Checksum']
    CG = np.zeros(maturity_1 * 12)
    VE = np.zeros(maturity_1 * 12)
    CM = np.zeros(maturity_1 * 12)
    GZ = np.zeros(maturity_1 * 12)
    TC = np.zeros(maturity_1 * 12)
    CZ = np.zeros(maturity_1 * 12)
    CA = np.zeros(maturity_1 * 12)
    CY = np.zeros(maturity_1 * 12)
    Total = np.zeros(maturity_1 * 12)
    Checksum = np.zeros(maturity_1 * 12)
    
    # Translate the former output from other function ( dataframe ) into numpy array
    balance_CG = balance.loc[:,'CG']
    balance_VE = balance.loc[:,'VE']
    balance_CM = balance.loc[:,'CM']
    GZ_Interest = principal_CF_Alloc.loc[:,'GZ_Interest']
    GZ_Accrued = principal_CF_Alloc.loc[:,'GZ_Accrued']
    balance_TC = balance.loc[:,'TC']
    CZ_Interest = principal_CF_Alloc.loc[:,'CZ_Interest']
    CZ_Accrued = principal_CF_Alloc.loc[:,'CZ_Accrued']
    balance_CA = balance.loc[:,'CA']
    balance_CY = balance.loc[:,'CY']
    
    # Calculation is based on formula from excel sheet
    for i in range(0, maturity_1 * 12):
        CG[i] = balance_CG[i]* tranche_coupon
        VE[i] = balance_VE[i] * tranche_coupon
        CM[i]= balance_CM[i] * tranche_coupon
        GZ[i] = GZ_Interest[i] - GZ_Accrued[i]
        TC[i] = balance_TC[i] * tranche_coupon
        CZ[i] = CZ_Interest[i] - CZ_Accrued[i]
        CA[i] = balance_CA[i] * tranche_coupon
        CY[i] = balance_CY[i] * tranche_coupon
        Total[i] = CG[i]+VE[i]+CM[i]+GZ[i]+TC[i]+CZ[i]+CA[i]+CY[i]
    
    temp = np.column_stack((CG,VE,CM,GZ,TC,CZ,CA,CY,Total,Checksum))
    Interest_df = pd.DataFrame(temp, columns =headers)
    return Interest_df    

def Pricing(principal,interest):
    # need to adjust index temporately for adjusting the HW1.py calculation

    cols = ['CG', 'VE', 'CM', 'GZ', 'TC', 'CZ', 'CA', 'CY']
    Pricing_df = principal[cols] + interest[cols]
    Pricing_df['Period'] = np.arange(1,len(principal)+1)
    return Pricing_df  

'''
pool_CF_1 = pool_CF_func(principal_1, monthly_interest_1, payment_number_1, PSA_1, maturity_1, age_1)
pool_CF_2 = pool_CF_func(principal_2, monthly_interest_2, payment_number_2, PSA_2, maturity_2, age_2)
summary_CF_df = summary_CF_func(pool_CF_1, pool_CF_2)
principal_CF_Alloc = Principal_CF_Alloc_func(summary_CF_df,tranche_coupon,CG,VE,CM,GZ,TC,CZ,CA,CY)
principal = Principal_func(principal_CF_Alloc)
balance = Balance_func(summary_CF_df,principal_CF_Alloc)
interest = Interest_func(balance,principal_CF_Alloc,summary_CF_df,tranche_coupon)
pricing = Pricing_func(principal,interest)
'''