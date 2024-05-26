
########## IMPORTING REQUIRED LIBRARIES. ###########

import os, pandas as pd, numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score
# from xgboost import XGBRegressor as XGBRegrwssor

########## READING FILES #########
train = pd.read_csv('/airqo-train.csv')
print(train.shape)

########### PREPROCESSING. #########
train['date']= pd.to_datetime(train.date)
train['dayofyear']= train.date.dt.dayofyear

regions ={'central':['Younde'],
          'eastern': ['Bujumbura', 'Nairobi', 'Kisumu', 'Gulu', 'Kampala'],
          'western': ['Accra', 'Lagos']
           }
def get_region(x):
  if x in regions['central']:
    return 'Central'
  elif x in regions['eastern']:
    return 'Eastern'
  elif x in regions['western']:
    return 'Western'

train['regions'] = train.city.apply(get_region)

zenith_cols = [col for col in train.columns if 'zenith' in col.split('_')]
#others = ['month', 'regions', 'hour', 'dayofyear']
#df1 = train[density_cols+others]
azimuth_cols = [col for col in train.columns if 'azimuth' in col.split('_')]
heights = [ col for col in train.columns for z in ['height', 'altitude', 'depth', 'temperature', 'albedo', 'pressure'] if z in col.split('_')]
#heights

catcols = [ 'month', 'hour', 'city', 'regions']
train['month'] = train.month.astype(int)
#train.month.unique()

onehot = OneHotEncoder(sparse_output=False, dtype=np.int32)
lblenc = LabelEncoder()
# train.loc[:, 'site_id'] = lblenc.fit_transform(train.site_id)

onehot.fit(train[catcols])
newcols = onehot.get_feature_names_out(catcols)
train.loc[:, newcols] = onehot.transform(train[catcols])
train.drop(catcols, axis=1, inplace=True)

########## MODELLING. ##########

df = train.drop(['id', 'site_id', 'date', 'country', 'site_latitude', 'site_longitude']+heights,axis=1)
# Dividing outliers greater than 300 by 10.
def reduce_pm2_5(x):
  if x>150:
    return x/10
  else:
    return x

df['pm2_5'] = df.pm2_5.apply(reduce_pm2_5)
#df.pm2_5.max()

xtrain, xvalid, ytrain, yvalid = train_test_split(df.drop('pm2_5', axis=1), df.pm2_5, test_size=0.2, random_state=524)
ytrain = ytrain.apply(reduce_pm2_5)
#xtrain.shape, xvalid.shape

regr = RFR(n_estimators=100, random_state=524, max_depth=11)

regr_pipe = Pipeline([
            ('imputer', KNNImputer(n_neighbors=1, weights='distance')),
            # ('standardizer', RobustScaler()),
            ('model', regr)])
regr_pipe.fit(xtrain, ytrain)
regr_pred = regr_pipe.predict(xtrain)
print('Train error:', mean_squared_error(ytrain, regr_pred, squared=False))

valid_pred = regr_pipe.predict(xvalid)
print('valid error:', mean_squared_error(yvalid, valid_pred, squared=False))


sns.scatterplot(y=ytrain, x=regr_pred, hue=xtrain['regions_Eastern'])
plt.figsave('train_residual.png')

sns.scatterplot(y=yvalid, x=valid_pred, hue=xvalid['regions_Eastern'],
             style=xvalid['month_2'])
plt.figsave('valid_residuals.png')
