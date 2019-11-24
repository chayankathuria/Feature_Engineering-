import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.stats as spstats

%matplotlib inline
mpl.style.reload_library()
mpl.style.use('classic')
mpl.rcParams['figure.facecolor'] = (1, 1, 1, 0)
mpl.rcParams['figure.figsize'] = [6.0, 4.0]
mpl.rcParams['figure.dpi'] = 100

# Binarization
'''
Binarization is nothing but converting numerical data into binary values of 0s and 1s based on
a threshold value. If the value is more than the threshold, it is assigned 1, else 0.

This can be done by simply putting a condition on the column and assigining 1 to values more than a threshold.
Binarizer in the preprocessing module does this task by using the threshold parameter,
'''

popsong_df = pd.read_csv('datasets/song_views.csv', encoding='utf-8')
popsong_df.head(10)

watched = np.array(popsong_df['listen_count']) 
watched[watched >= 1] = 1
popsong_df['watched'] = watched
popsong_df.head(10)

# Now using the binarizer of sklearn

from sklearn.preprocessing import Binarizer

bn = Binarizer(threshold=0.9)
pd_watched = bn.transform([popsong_df['listen_count']])[0]
popsong_df['pd_watched'] = pd_watched
popsong_df.head(11)

# Rounding
'''
Rounding the values to nearest 10s 1or 100s 
'''

items_popularity = pd.read_csv('datasets/item_popularity.csv', encoding='utf-8')
items_popularity

items_popularity['popularity_scale_10'] = np.array(np.round((items_popularity['pop_percent'] * 10)), dtype='int')
items_popularity['popularity_scale_100'] = np.array(np.round((items_popularity['pop_percent'] * 100)), dtype='int')
items_popularity

# Interactions
'''
The polynomial regression adds interaction terms like x*y or 2*x*y etc. This interaction term can be a better 
feature as it might have better correaltion with the target
'''

atk_def = poke_df[['Attack', 'Defense']]
atk_def.head()

from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
res = pf.fit_transform(atk_def)
res

intr_features = pd.DataFrame(res, columns=['Attack', 'Defense', 'Attack^2', 'Attack x Defense', 'Defense^2'])
intr_features.head(5)  

'''
Attack	Defense	Attack^2	Attack x Defense	Defense^2
0	49.0	49.0	  2401.0  	     2401.0	       2401.0
1	62.0	63.0	  3844.0	       3906.0	       3969.0
2	82.0	83.0	  6724.0	       6806.0	       6889.0
3	100.0	123.0	  10000.0	       12300.0	     15129.0
4	52.0	43.0	  2704.0	       2236.0	       1849.0
'''
# Mathematical Transformations
## Log Transformation and Box-Cox plots

'''
Dealing with skewed data can be cumbersome and we need to transform the data in order to make the distribution normal.
Log transformation and Box-Cox plots are 2 such techniques
'''
fcc_survey_df['Income_log'] = np.log((1+ fcc_survey_df['Income'])) # Here 1 is added in order to avoid log(0) as it is not defined. 
fcc_survey_df[['ID.x', 'Age', 'Income', 'Income_log']].iloc[4:9]   # Adding 1 will not change the distribution and just shift the data

'''
                  ID.x	             Age	Income	Income_log
4	9368291c93d5d5f5c8cdb1a575e18bec	20.0	6000.0	8.699681
5	dd0e77eab9270e4b67c19b0d6bbf621b	34.0	40000.0	10.596660
6	7599c0aa0419b59fd11ffede98a3665d	23.0	32000.0	10.373522
7	6dff182db452487f07a47596f314bddc	35.0	40000.0	10.596660
8	9dc233f8ed1c6eb2432672ab4bb39249	33.0	80000.0	11.289794
'''

# Box–Cox transform
'''
A Box Cox transformation is a way to transform non-normal dependent variables into a normal shape. 
Normality is an important assumption for many statistical techniques; 
if your data isn’t normal, applying a Box-Cox means that you are able to run a broader number of tests.
'''

# get optimal lambda value from non null income values
income = np.array(fcc_survey_df['Income'])
income_clean = income[~np.isnan(income)]
l, opt_lambda = spstats.boxcox(income_clean)
print('Optimal lambda value:', opt_lambda)

# Optimal lambda value: 0.117991239456

fcc_survey_df['Income_boxcox_lambda_0'] = spstats.boxcox((1+fcc_survey_df['Income']), 
                                                         lmbda=0)
fcc_survey_df['Income_boxcox_lambda_opt'] = spstats.boxcox(fcc_survey_df['Income'], 
                                                           lmbda=opt_lambda)
fcc_survey_df[['ID.x', 'Age', 'Income', 'Income_log', 
               'Income_boxcox_lambda_0', 'Income_boxcox_lambda_opt']].iloc[4:9]

'''
      ID.x	                        Age	  Income	Income_log Income_boxcox_lambda_0	  Income_boxcox_lambda_opt
4	9368291c93d5d5f5c8cdb1a575e18bec	20.0	6000.0	8.699681	      8.699681	                  15.180668
5	dd0e77eab9270e4b67c19b0d6bbf621b	34.0	40000.0	10.596660	      10.596660                 	21.115342
6	7599c0aa0419b59fd11ffede98a3665d	23.0	32000.0	10.373522	      10.373522                 	20.346420
7	6dff182db452487f07a47596f314bddc	35.0	40000.0	10.596660	      10.596660                 	21.115342
8	9dc233f8ed1c6eb2432672ab4bb39249	33.0	80000.0	11.289794	      11.289794                  	23.637131
'''



