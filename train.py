
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
train = pd.read_csv(path +'/airqo-train.csv')
print(train.shape)
