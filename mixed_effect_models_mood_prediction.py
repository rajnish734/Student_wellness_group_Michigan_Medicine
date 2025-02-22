# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:24:46 2024

Apply mixed effect model to predict MOOD from sleep details and total step count.

@author: rajnishk
"""


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.dimred import CORE
from scipy.linalg import eigh
from matplotlib.patches import Ellipse
from scipy.stats.distributions import chi2
from statsmodels.nonparametric.smoothers_lowess import lowess
# import numpy as np
# import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf



df_sleep = pd.read_csv(r'C:\Users\rajnishk\OneDrive - Michigan Medicine\Documents\Student Wellness Dataset\student_wellness_sleep_details.csv',
                 parse_dates=["SLEEP_DATE","SLEEP_START_DATE", "SLEEP_END_DATE"],index_col=0)

df_MOOD = pd.read_csv(r'C:\Users\rajnishk\OneDrive - Michigan Medicine\Documents\Student Wellness Dataset\df_MOOD_SWG.csv', 
                     parse_dates=["METRIC_START_DATE", "METRIC_END_DATE"],index_col=0)


df_MOOD.rename(columns={'METRIC_START_DATE': 'SLEEP_DATE'}, inplace=True) # So that I can merge two dataframes. 

#==================== Collecting the sleep variables together =========================================================================

sleep_vars = ['ASLEEP_VALUE', 'INBED_VALUE', 'DEEP_MIN', 'DEEP_COUNT', 'LIGHT_MIN', 'LIGHT_COUNT', 
              'REM_MIN', 'REM_COUNT', 'WAKE_MIN', 'WAKE_COUNT', 'ASLEEP_MIN', 'ASLEEP_COUNT', 
              'AWAKE_COUNT', 'AWAKE_MIN', 'RESTLESS_COUNT', 'RESTLESS_MIN']
v = df_sleep[sleep_vars].isna().sum(0)
v = v[v < 500].index.tolist()
vv = ["STUDY_PRTCPT_ID", "SLEEP_START_DATE", "SLEEP_END_DATE", "SLEEP_DATE"] + v
dx = df_sleep[vv].copy()
dx["YEARDAY"] = dx["SLEEP_DATE"].dt.dayofyear
dx["DAYOFWEEK"] = dx["SLEEP_DATE"].dt.dayofweek
dx["SLEEP_START_TIME"] = (dx["SLEEP_START_DATE"] - dx["SLEEP_START_DATE"].dt.normalize()) / pd.Timedelta(hours=1)
dx["SLEEP_END_TIME"] = (dx["SLEEP_END_DATE"] - dx["SLEEP_END_DATE"].dt.normalize()) / pd.Timedelta(hours=1)
dx["SLEEP_START_SIN"] = np.sin(2*np.pi*dx["SLEEP_START_TIME"]/24)
dx["SLEEP_START_COS"] = np.cos(2*np.pi*dx["SLEEP_START_TIME"]/24)
dx["SLEEP_END_SIN"] = np.sin(2*np.pi*dx["SLEEP_END_TIME"]/24)
dx["SLEEP_END_COS"] = np.cos(2*np.pi*dx["SLEEP_END_TIME"]/24)
dx["YEARDAY_SIN"] = np.sin(2*np.pi*dx["YEARDAY"]/366)
dx["YEARDAY_COS"] = np.cos(2*np.pi*dx["YEARDAY"]/366)

dx["STUDYDAY"] = dx["YEARDAY"] - dx.groupby("STUDY_PRTCPT_ID")["YEARDAY"].transform(np.min)
# dx = dx.drop(columns=["SLEEP_START_DATE", "SLEEP_END_DATE", "SLEEP_START_TIME", "SLEEP_END_TIME", "SLEEP_DATE"])
dx = dx.dropna()

df_MOOD_sleep_polar = pd.merge(dx,df_MOOD, on = ['STUDY_PRTCPT_ID','SLEEP_DATE'], how = 'inner' )
df_MOOD_sleep_polar_MOOD_NZ = df_MOOD_sleep_polar[df_MOOD_sleep_polar.MOOD !=0]   # Lots of data goes away. MOOD ==0 values are dropped.

check_2_sleep_vars = df_MOOD_sleep_polar_MOOD_NZ.ASLEEP_VALUE - df_MOOD_sleep_polar_MOOD_NZ.SLEEP_COUNT # There are a lot of non zeros
check_2_sleep_vars_NZ = check_2_sleep_vars[check_2_sleep_vars!=0]

# Normalizing the variables considered for analysis

columns_for_norm = ['ASLEEP_VALUE', 'INBED_VALUE', 'DEEP_MIN', 'DEEP_COUNT', 'LIGHT_MIN',
       'LIGHT_COUNT', 'REM_MIN', 'REM_COUNT', 'WAKE_MIN', 'WAKE_COUNT','STEP_COUNT', 'SLEEP_COUNT', 'MOOD']


# Normalize the sleep detail variables, total step variable, and MOOD. I am normalizing the MOOD because it is a part of factor analysis, and it might be interesting to see the appearance of MOOD in the vicinity of other sleep variables. 

df_MOOD_sleep_polar_MOOD_NZ_norm = df_MOOD_sleep_polar_MOOD_NZ.copy()
df_MOOD_sleep_polar_MOOD_NZ_norm[columns_for_norm] = df_MOOD_sleep_polar_MOOD_NZ_norm[columns_for_norm]- df_MOOD_sleep_polar_MOOD_NZ_norm[columns_for_norm].mean(0)
df_MOOD_sleep_polar_MOOD_NZ_norm[columns_for_norm] = df_MOOD_sleep_polar_MOOD_NZ_norm[columns_for_norm]/df_MOOD_sleep_polar_MOOD_NZ_norm[columns_for_norm].std(0)

np.random.seed(123)
#=================================================================================================================================================================
model_first = smf.mixedlm('MOOD ~ ASLEEP_VALUE + INBED_VALUE + DEEP_MIN + DEEP_COUNT + LIGHT_MIN + LIGHT_COUNT +REM_MIN  +REM_COUNT +WAKE_MIN + WAKE_COUNT +STEP_COUNT  +SLEEP_COUNT', df_MOOD_sleep_polar_MOOD_NZ_norm, groups = df_MOOD_sleep_polar_MOOD_NZ_norm['STUDY_PRTCPT_ID'])
result_first = model_first.fit()

print(result_first.summary())
#=================================================================================================================================================================
model_second = smf.mixedlm('MOOD ~ ASLEEP_VALUE + INBED_VALUE + DEEP_MIN + DEEP_COUNT + LIGHT_MIN + LIGHT_COUNT +REM_MIN  +REM_COUNT +WAKE_MIN + WAKE_COUNT +STEP_COUNT + SLEEP_END_TIME  +SLEEP_COUNT', df_MOOD_sleep_polar_MOOD_NZ_norm, groups = df_MOOD_sleep_polar_MOOD_NZ_norm['STUDY_PRTCPT_ID'])
result_second = model_second.fit()

print(result_second.summary())
#=================================================================================================================================================================
model_third = smf.mixedlm('MOOD ~ ASLEEP_VALUE + INBED_VALUE + SLEEP_START_TIME+ DEEP_MIN + DEEP_COUNT + LIGHT_MIN + LIGHT_COUNT +REM_MIN  +REM_COUNT +WAKE_MIN + WAKE_COUNT +STEP_COUNT + SLEEP_END_TIME  +SLEEP_COUNT', df_MOOD_sleep_polar_MOOD_NZ_norm, groups = df_MOOD_sleep_polar_MOOD_NZ_norm['STUDY_PRTCPT_ID'])
result_third = model_third.fit()

print(result_third.summary())
#=================================================================================================================================================================
print("Changing the list of input variables changes the significance level of some of the independent variables. I will make sense of the reason later.")

# Include day of the week as well
model_fourth = smf.mixedlm('MOOD ~ ASLEEP_VALUE + INBED_VALUE + SLEEP_START_TIME+ DEEP_MIN + DEEP_COUNT + LIGHT_MIN + LIGHT_COUNT +REM_MIN  +REM_COUNT +WAKE_MIN + WAKE_COUNT +STEP_COUNT + DAYOFWEEK+ SLEEP_END_TIME  +SLEEP_COUNT', df_MOOD_sleep_polar_MOOD_NZ_norm, groups = df_MOOD_sleep_polar_MOOD_NZ_norm['STUDY_PRTCPT_ID'])
result_fourth = model_fourth.fit()

print(result_fourth.summary())

# Grouping by day of week instead of students
model_fifth = smf.mixedlm('MOOD ~ ASLEEP_VALUE + INBED_VALUE + DEEP_MIN + DEEP_COUNT + LIGHT_MIN + LIGHT_COUNT +REM_MIN  +REM_COUNT +WAKE_MIN + WAKE_COUNT +STEP_COUNT  +SLEEP_COUNT', df_MOOD_sleep_polar_MOOD_NZ_norm, groups = df_MOOD_sleep_polar_MOOD_NZ_norm['DAYOFWEEK'])
result_fifth = model_fifth.fit()

print(result_fifth.summary())

model_sixth = smf.mixedlm('MOOD ~ ASLEEP_VALUE + INBED_VALUE + DEEP_MIN + DEEP_COUNT + LIGHT_MIN + LIGHT_COUNT +REM_MIN  +REM_COUNT +WAKE_MIN + WAKE_COUNT +STEP_COUNT + SLEEP_END_TIME  +SLEEP_COUNT', df_MOOD_sleep_polar_MOOD_NZ_norm, groups = df_MOOD_sleep_polar_MOOD_NZ_norm['DAYOFWEEK'])
result_sixth = model_sixth.fit()

print(result_sixth.summary())

model_7 = smf.mixedlm('MOOD ~ ASLEEP_VALUE + INBED_VALUE + SLEEP_START_TIME+ DEEP_MIN + DEEP_COUNT + LIGHT_MIN + LIGHT_COUNT +REM_MIN  +REM_COUNT +WAKE_MIN + WAKE_COUNT +STEP_COUNT + SLEEP_END_TIME  +SLEEP_COUNT', df_MOOD_sleep_polar_MOOD_NZ_norm, groups = df_MOOD_sleep_polar_MOOD_NZ_norm['DAYOFWEEK'])
result_7 = model_7.fit()

print(result_7.summary())

# Multiple random effects. Following reference has explanation. 
# https://www.statsmodels.org/stable/examples/notebooks/generated/mixed_lm_example.html

# Adding random effects of both person as well as weak days to see the effect on MOOD

# col_4_2_R_effects = ['STUDY_PRTCPT_ID', 'SLEEP_START_DATE', 'SLEEP_END_DATE', 'SLEEP_DATE',
#        'ASLEEP_VALUE', 'INBED_VALUE', 'DEEP_MIN', 'DEEP_COUNT', 'LIGHT_MIN',
#        'LIGHT_COUNT', 'REM_MIN', 'REM_COUNT', 'WAKE_MIN', 'WAKE_COUNT',
#         'DAYOFWEEK', 'SLEEP_START_TIME', 'SLEEP_END_TIME',
       
#        'YEARDAY_SIN', 'YEARDAY_COS', 'STUDYDAY', 
#         'STEP_COUNT', 'SLEEP_COUNT', 'MOOD', ]

# df_2_random_effects = df_MOOD_sleep_polar_MOOD_NZ_norm[col_4_2_R_effects]

# model_8 = smf.mixedlm('MOOD ~ ASLEEP_VALUE + INBED_VALUE + DEEP_MIN + DEEP_COUNT + LIGHT_MIN + LIGHT_COUNT +REM_MIN  +REM_COUNT +WAKE_MIN + WAKE_COUNT +STEP_COUNT  +SLEEP_COUNT', df_MOOD_sleep_polar_MOOD_NZ_norm, groups = df_MOOD_sleep_polar_MOOD_NZ_norm['STUDY_PRTCPT_ID'], re_formula = "" )

#==============================================================================================================================================================
# Products of variables -- confirm with Dr. Shedden whether normalization comes first or after. 




