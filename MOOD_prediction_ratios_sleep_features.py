#!/usr/bin/env python
# coding: utf-8

# In this code, we will calculate new features as a combinarion of ratios between sleep details along with the activity count, and explore the relationship of these newly engineered features with MOOD of the participant. 

# In[1]:


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


# In[2]:


df_sleep = pd.read_csv(r'C:\Users\rajnishk\OneDrive - Michigan Medicine\Documents\Student Wellness Dataset\student_wellness_sleep_details.csv',
                 parse_dates=["SLEEP_DATE","SLEEP_START_DATE", "SLEEP_END_DATE"],index_col=0)

df_MOOD = pd.read_csv(r'C:\Users\rajnishk\OneDrive - Michigan Medicine\Documents\Student Wellness Dataset\df_MOOD_SWG.csv', 
                     parse_dates=["METRIC_START_DATE", "METRIC_END_DATE"],index_col=0)


# In[3]:


df_MOOD.rename(columns={'METRIC_START_DATE': 'SLEEP_DATE'}, inplace=True) # So that I can merge two dataframes. 


# In[4]:


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

# #===============================================================================================
# dx["SLEEP_START_TIME"] = (dx["SLEEP_START_DATE"] - dx["SLEEP_START_DATE"].dt.normalize()) / pd.Timedelta(hours=1)
# plt.scatter(dx["SLEEP_START_TIME"],dx["SLEEP_START_SIN"] )
# #===============================================================================================


dx["STUDYDAY"] = dx["YEARDAY"] - dx.groupby("STUDY_PRTCPT_ID")["YEARDAY"].transform(np.min)
# dx = dx.drop(columns=["SLEEP_START_DATE", "SLEEP_END_DATE", "SLEEP_START_TIME", "SLEEP_END_TIME", "SLEEP_DATE"])
dx = dx.dropna()
dx.columns


# In[5]:


df_MOOD_sleep_polar = pd.merge(dx,df_MOOD, on = ['STUDY_PRTCPT_ID','SLEEP_DATE'], how = 'inner' )
df_MOOD_sleep_polar_MOOD_NZ = df_MOOD_sleep_polar[df_MOOD_sleep_polar.MOOD !=0]   # Lots of data goes away.


# This is just to get everyone's attention. Check if SLEEP_COUNT and ASLEEP_VALUE are the same. 

# In[6]:


abcd = df_MOOD_sleep_polar_MOOD_NZ.ASLEEP_VALUE - df_MOOD_sleep_polar_MOOD_NZ.SLEEP_COUNT
abcd[abcd!=0]


# In[7]:


# Normalizing the variables considered for analysis


# In[8]:


columns_for_norm = ['ASLEEP_VALUE', 'INBED_VALUE', 'DEEP_MIN', 'DEEP_COUNT', 'LIGHT_MIN',
       'LIGHT_COUNT', 'REM_MIN', 'REM_COUNT', 'WAKE_MIN', 'WAKE_COUNT','STEP_COUNT', 'SLEEP_COUNT', 'MOOD']


# Normalize the sleep detail variables, total step variable, and MOOD. I am normalizing the MOOD because it is a part of factor analysis, and it might be interesting to see the appearance of MOOD in the vicinity of other sleep variables. 

df_MOOD_sleep_polar_MOOD_NZ_norm = df_MOOD_sleep_polar_MOOD_NZ.copy()
df_MOOD_sleep_polar_MOOD_NZ_norm[columns_for_norm] = df_MOOD_sleep_polar_MOOD_NZ_norm[columns_for_norm]- df_MOOD_sleep_polar_MOOD_NZ_norm[columns_for_norm].mean(0)
df_MOOD_sleep_polar_MOOD_NZ_norm[columns_for_norm] = df_MOOD_sleep_polar_MOOD_NZ_norm[columns_for_norm]/df_MOOD_sleep_polar_MOOD_NZ_norm[columns_for_norm].std(0)
df_MOOD_sleep_polar_MOOD_NZ_norm


# In[9]:


# Calculate the ratio features as new features for factor analysis and machine learning for further analysis


# In[10]:


features_for_ratios = ['ASLEEP_VALUE',  'INBED_VALUE',  'DEEP_MIN',  'DEEP_COUNT',  'LIGHT_MIN',  'LIGHT_COUNT',
 'REM_MIN',  'REM_COUNT',  'WAKE_MIN',  'WAKE_COUNT',  'STEP_COUNT',  'SLEEP_COUNT']
feature_remaining = ['MOOD']


# In[11]:


# features_for_ratios


# In[12]:


epsilon = 1e-10  # to avoid division by zero


# In[13]:


X = df_MOOD_sleep_polar_MOOD_NZ_norm[features_for_ratios]
X_remain_feature = df_MOOD_sleep_polar_MOOD_NZ_norm[feature_remaining]


# In[14]:


X_forward_ratios = X.copy()
X_reverse_ratios = X.copy()


# In[15]:


X_reverse_ratios


# In[16]:


for i,feature1 in enumerate(features_for_ratios):
    for feature2 in features_for_ratios[i+1:]:
        X_forward_ratios[f'ratio_{feature1}_{feature2}'] = X_forward_ratios[feature1] / (X_forward_ratios[feature2] + epsilon)


# In[17]:


for i,feature1 in enumerate(features_for_ratios):
    for feature2 in features_for_ratios[i+1:]:
        X_reverse_ratios[f'ratio_{feature2}_{feature1}'] = X_reverse_ratios[feature2] / (X_reverse_ratios[feature1] + epsilon)


# In[25]:


df_final_forward_ratio = pd.concat([X_forward_ratios, X_remain_feature],axis=1)
df_final_forward_ratio


# In[26]:


df_final_reverse_ratio = pd.concat([X_reverse_ratios, X_remain_feature],axis=1)
df_final_reverse_ratio


# In[ ]:





# # # PCA/biplots

# In[37]:


# EITHER PICK THIS PAIR
# dx = df_final_forward_ratio
# va = df_final_forward_ratio.columns  


# OR THIS PAIR 
dx = df_final_reverse_ratio
va = df_final_reverse_ratio.columns


# In[ ]:





# In[38]:


def plot_eigs(eigs):
    jj = np.arange(1, len(eigs) + 1)
    ii = np.flatnonzero(eigs >= 1e-10)
    plt.clf()
    plt.grid(True)
    plt.plot(np.log(jj[ii]), np.log(eigs[ii]), "-o")
    # plt.plot((jj[ii]), (eigs[ii]), "-o")
    
    plt.ylabel("Log eigenvalue", size=15)
    plt.xlabel("Log position", size=15)
    plt.show()


# In[39]:


def varimax(X, eps=1e-05, maxiter=1000):
    G = X.copy()
    m, p = G.shape
    Q = np.eye(p)
    if p < 2: 
        return G, Q
    d = 0.0
    for i in range(maxiter):
        z = np.dot(G, Q)
        cs = (z**2).sum(0)
        B = np.dot(G.T, z**3 - z * (cs / m))
        u, s, vt = np.linalg.svd(B, 0)
        Q = np.dot(u, vt)
        dlast = d
        d = s.sum()
        if d < dlast * (1 + eps): 
            break
            
    G = np.dot(G, Q)
    return G, Q


# In[40]:


def ellipse_plot(U, idx, j0, j1):
    r2 = chi2(2).ppf(0.95)
    rr = np.sqrt(r2)
    dx = pd.DataFrame({"u0": U[:, 0], "u1": U[:, 1], "id": idx})
    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.grid(True)
    plt.plot(dx["u0"], dx["u1"], "o", color="black", alpha=0.05)
    for (ky,dg) in dx.groupby("id"):
        if dg.shape[0] > 10:
            m0, m1 = dg["u0"].mean(), dg["u1"].mean()
            C = np.cov(dg["u0"], dg["u1"])
            a, b = eigh(C)
            plt.plot([m0], [m1], "o", color="blue", ms=8, alpha=0.2)
            ang = np.arctan2(b[1, 1], b[0, 1])*360/(2*np.pi)
            E = Ellipse(xy=[m0,m1], width=rr*np.sqrt(a[1]), height=rr*np.sqrt(a[0]), fc="none", ec="black", angle=ang)
            plt.gca().add_artist(E)
    plt.xlabel("Component %d" % j0, size=17)
    plt.ylabel("Component %d" % j1, size=17)
    plt.show()


# In[41]:


def biplot(dx, va, j=0, k=1, d=5, rotate=False, scree=False, ellipses=False):
    assert d > max(j, k)
    dx = dx.copy()
    X = np.asarray(dx[va])
    n, p = X.shape
    X -= X.mean()
    X -= X.mean(0)
    u, s, vt = np.linalg.svd(X, 0)
    v = vt.T
    if scree:
        plot_eigs(s)
    uu = u[:, 0:d]
    vv = v[:, 0:d]
    ss = s[0:d]
    if rotate:
        uu, Ru = varimax(uu)
        uu *= np.sqrt(n)
        vv, Rv = varimax(vv)
        vv *= np.sqrt(p)
        B = np.dot(Ru.T * ss, Rv) / np.sqrt(n*p)
    else:
        uu *= (ss**0.5)
        vv *= (ss**0.5)
        B = np.eye(d)
        
    ff = np.sqrt(X.shape[0] / X.shape[1]) # May need to adjust this
    v /= ff
    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.grid(True)
    plt.plot(uu[:, j], uu[:, k], "o", color="black", alpha=0.05)
    for i in range(v.shape[0]):
        
        plt.annotate(va[i], xy=(0, 0), xytext=(vv[i, j], vv[i, k]), 
                     arrowprops=dict(color='orange', arrowstyle="<-"), color="purple", size=9)
    plt.xlabel("Component %d" % j)
    plt.ylabel("Component %d" % k)
    plt.show()
    
    if ellipses:
        ellipse_plot(uu[:, [j, k]], dx["STUDY_PRTCPT_ID"], j,k)
print("va is %s" %va)        
biplot(dx, va, 0, 1, d=5, rotate=False, scree=True, ellipses=True)
biplot(dx, va, 2, 3, d=5, rotate=False, ellipses=True)
biplot(dx, va, 0, 1, d=5, rotate=True, ellipses=True)
biplot(dx, va, 2, 3, d=5, rotate=True, ellipses=True)


# In[ ]:




