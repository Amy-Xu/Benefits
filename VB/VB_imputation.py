
# coding: utf-8

# This script imputes both the participation and dollar benefit amount for the VA benefit programs to match the aggregates obtained from the United States Department of Veterans Affairs (VA) statistics (https://www.va.gov/vetdata/; expenditure). 
# 
# This current version of imputation is based on the CPS March Supplement and the Veteran's expenditure report available from the website linked above for the corresponding year. Please refer to the documentation in the same folder for more details on methodology and assumptions. The output this script is an individual level dataset that contains CPS personal level ID (PERIDNUM), individual participation indicator (vb_participation, 0 - not a recipient, 1 - current recipient on file, 2 - imputed recipient), and benefit amount.
# 
# One assumption made for participants in the imputation is that 48% of veterans use the benefit programs, according to 2016 statistics. https://www.va.gov/vetdata/docs/QuickFacts/VA_Utilization_Profile.PDF
# 
# Input command line:
# python VB_imputation.py CPS_FILENAME YEAR OUTPUT_FILENAME
# 
# (Inpute data file should be placed in the working directory)


import pandas as pd
from pandas import DataFrame
import numpy as np
import random
import statsmodels.discrete.discrete_model as sm
import sys


# assumptions and constants to use
DELTA = 10e-8 # constant added to avoid zero denomination
Portion = 0.48 # proportion of veteran's who uses at least one program

# read the arguments from command lines
if len(sys.argv) != 4:
    print 'Please follow the format: python VB_imputation.py CPS_FILENAME YEAR OUTPUT_FILENAME'
else:
    CPS_FILENAME = sys.argv[1]
    YEAR = sys.argv[2]
    OUTPUT_FILENAME = sys.argv[3]


# input datafile
CPS_dataset = pd.read_csv(CPS_FILENAME)

# create the dummies first - orginal variable have multiple values
VB = DataFrame(CPS_dataset[['vet_yn', 'a_sex', # veteran status & gender
                            'pedisdrs','pedisear', 'pediseye', 'pedisout', 'pedisphy', 'pedisrem', # disability
                            'paw_yn', 'wc_yn', 'uc_yn', 'ss_yn', 'sur_yn', # other welfare/transfer
                            'hed_yn','hcsp_yn', 'hfoodsp','mcaid','mcare']])
VB.replace([-1, 2], 0, inplace=True)


# add numerical values 
num_vars = ['marsupwt', 'fh_seq', 'gestfips', 'peridnum','vet_val', 'a_age',
            'wsal_val', 'semp_val', 'frse_val']
VB[num_vars] =CPS_dataset[num_vars]
VB.rename(index=str, columns = {'vet_val':'vbvalue'}, inplace=True)

# income is approximated as wage, self-employed and farm income
VB['income'] = VB.wsal_val + VB.semp_val + VB.frse_val

# Potential recipients also include the family members of veterans
vet_family = DataFrame(VB.groupby('fh_seq', as_index=False)['vet_yn'].sum())
vet_family.rename(index=str, columns = {'vet_yn': 'vet_family'}, inplace=True)
VB = VB.merge(vet_family, how='left', on='fh_seq')


# Logit Regression for Propensity Score
VB['intercept'] = np.ones(len(VB))
model = sm.Logit(endog=VB.vet_yn, 
                 exog=VB[['a_age','a_sex','income', 'vet_family',
                          'pedisdrs','pedisear', 'pediseye', 'pedisout', 'pedisphy', 'pedisrem',
                          'paw_yn','wc_yn','ss_yn', 'uc_yn', 'sur_yn',
                          'hed_yn','hcsp_yn', 'hfoodsp','mcaid','mcare','intercept']]).fit()
# print model.summary()
probs = model.predict()

# Import administrative data
admin = pd.read_csv('VB_administrative.csv',
                    dtype={'Total VB population': np.float,
                           'Average benefits': np.float, 
                           'Total benefits': np.float, 
                           'Medical care': np.float})
admin.index = admin.Fips


# CPS total benefits and Administrative total benefits
state_benefit = {}
state_recipients = {}
for state in admin.Fips:
    this_state = (VB.gestfips==state)
    CPS_totalb = (VB.vbvalue * VB.marsupwt)[this_state].sum()
    admin_totalb =  admin['Total benefits'][state]
    CPS_totaln = VB.marsupwt[this_state&VB.vet_yn==1].sum()
    admin_totaln =  admin["Total veteran"][state]

    temp = [admin.State[state], CPS_totalb, admin_totalb, CPS_totaln, admin_totaln]
    state_benefit[state] = temp
    
pre_augment_benefit = DataFrame(state_benefit).transpose()
pre_augment_benefit.columns = ['State', 'CPS total benefits','Admin total benefits',
                               'CPS total individual recipients','Admin total individual recipients']
# export upon request
# pre_augment_benefit.to_csv('VB-pre-blow-up.csv')


# Imputation
# caculate difference of Veteran's Benefit targets and CPS aggregates
diff = {'Fips':[],'Difference in Population':[],
        'Mean Benefit':[],'CPS Population':[],'VA Population':[]}
diff['Fips'] = admin.Fips
current = (VB.vet_yn==1)

for FIPS in admin.Fips:
    
        this_state = (VB.gestfips==FIPS)
        
        current_tots = VB.marsupwt[current&this_state].sum()
        diff['CPS Population'].append(current_tots)
        
        valid_num = VB.marsupwt[current&this_state].sum() + DELTA
        current_mean = ((VB.vbvalue * VB.marsupwt)[current&this_state].sum())/valid_num
        diff['Mean Benefit'].append(current_mean)
        
        
        Total_participate = float(admin["Total veteran"][admin.Fips == FIPS]) * Portion
        diff['VA Population'].append(Total_participate)
        diff['Difference in Population'].append(Total_participate - current_tots)


d = DataFrame(diff)
d = d[['Fips', 'Mean Benefit', 'Difference in Population', 'CPS Population', 'VA Population']]

# d.to_csv('recipients.csv', index=False)

# Impute participants
VB['impute'] = np.zeros(len(VB))
VB['vb_impute'] = np.zeros(len(VB))

non_current = (VB.vet_yn==0)
current = (VB.vet_yn==1)
random.seed()

for FIPS in admin.Fips:

        if d['Difference in Population'][FIPS] < 0:
            continue
        else:
            this_state = (VB.gestfips==FIPS)
            not_imputed = (VB.impute==0) & (VB.vet_family>0)
            pool_index = VB[this_state&not_imputed&non_current].index
            
            pool = DataFrame({'weight': VB.marsupwt[pool_index], 'prob': probs[pool_index]},
                            index=pool_index)
            pool = pool.sort_values(by='prob', ascending=False)
            pool['cumsum_weight'] = pool['weight'].cumsum()
            pool['distance'] = abs(pool.cumsum_weight-d['Difference in Population'][FIPS])
            min_index = pool.sort_values(by='distance')[:1].index
            min_weight = int(pool.loc[min_index].cumsum_weight)
            pool['impute'] = np.where(pool.cumsum_weight<=min_weight+10 , 1, 0)
            VB.impute[pool.index[pool['impute']==1]] = 1
            VB.vb_impute[pool.index[pool['impute']==1]] = admin['Average benefits'][FIPS]


#Adjustment ratio
results = {}

imputed = (VB.impute == 1)
has_val = (VB.vbvalue != 0)
no_val = (VB.vbvalue == 0)

for FIPS in admin.Fips:
    this_state = (VB.gestfips==FIPS)
    
    current_total = (VB.vbvalue * VB.marsupwt)[this_state].sum() 
    imputed_total = (VB.vb_impute * VB.marsupwt)[this_state&imputed].sum()
    on_file = current_total + imputed_total

    admin_total = admin['Total benefits'][FIPS]
    
    adjust_ratio = admin_total / on_file
    this_state_num = [admin['State'][FIPS], on_file, admin_total, adjust_ratio]
    results[FIPS] = this_state_num
    

    VB.vb_impute = np.where(has_val&this_state, VB.vbvalue * adjust_ratio, VB.vb_impute)
    VB.vb_impute = np.where(no_val&this_state, VB.vb_impute * adjust_ratio, VB.vb_impute)

VB["vb_participation"] = np.zeros(len(VB))
VB["vb_participation"] = np.where(VB.impute==1, 2, 0)#Augmented
VB["vb_participation"] = np.where(has_val, 1, VB.vb_participation)#CPS 


r = DataFrame(results).transpose()
r.columns=['State', 'Imputed', 'Admin', 'adjust ratio']
#r.to_csv('amount.csv', index=False)


# Impute Medical care
medical = {}
medicalcare = {}
for FIPS in admin.Fips:
    this_state = (VB.gestfips==FIPS)
    medical[FIPS] = admin['Medical care'][FIPS] / (VB.marsupwt[VB.vb_participation==1][this_state].sum())
    VB.vb_impute = np.where((VB.vb_participation==1) & (this_state), VB.vb_impute + medical[FIPS], VB.vb_impute)
    medicalcare[FIPS] = [admin['State'][FIPS], medical[FIPS]]


VB.to_csv(OUTPUT_FILENAME,
         columns=['peridnum','vb_participation', 'vb_impute', 'probs'],
         index = False)


