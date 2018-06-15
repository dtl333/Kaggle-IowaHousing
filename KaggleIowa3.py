import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt 
import sklearn
from sklearn.preprocessing import StandardScaler, CategoricalEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV

#Import our Iowas housing training dataset and testing dataset
train = pd.read_csv('/Users/dylanloughlin/Desktop/PythonTUT/IowaHousing/train.csv')
housing_test = pd.read_csv('/Users/dylanloughlin/Desktop/PythonTUT/IowaHousing/test.csv')
housing_train_ = train.drop('SalePrice', axis=1)
housing_train = pd.concat([housing_train_, housing_test])
#housing_labels = train['SalePrice']
#print(housing_train.columns) #-- View our attributes(SalePrice is our target)

#STEP 1: LETS CLEANUP OUR DATA

#First lets deal with our NaN values in the datasets. We will use df_all so that our transformations will occur to both
cols_with_na = housing_train.isnull().sum()
cols_with_na = cols_with_na[cols_with_na>0]
#print(cols_with_na.sort_values(ascending=False))

#Lets define a variable with the features(columns) we want to fill
cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2']
#Since the NaN in these columns simply means the house does not have that feature lets fill with 'None'
for col in cols_fillna:
    housing_train[col].fillna('None', inplace=True)

#For GarageYrBlt, NaN means the garage was built with the original house, so replace with YearBuilt
housing_train.loc[housing_train.GarageYrBlt.isnull(), 'GarageYrBlt'] = housing_train.loc[housing_train.GarageYrBlt.isnull(), 'YearBuilt']

#For MasVnrArea(Masonry Veneer) -- fill with 0
housing_train.MasVnrArea.fillna(0,inplace=True)

#For houses with no basements/garage, fill all basement/garage features with 0
housing_train.BsmtFullBath.fillna(0,inplace=True)
housing_train.BsmtHalfBath.fillna(0,inplace=True)
housing_train.BsmtFinSF1.fillna(0,inplace=True)
housing_train.BsmtFinSF2.fillna(0,inplace=True)
housing_train.BsmtUnfSF.fillna(0,inplace=True)
housing_train.TotalBsmtSF.fillna(0,inplace=True)
housing_train.GarageArea.fillna(0,inplace=True)
housing_train.GarageCars.fillna(0,inplace=True)

#Now to deal with LotFrontage, to fill these NaN values we use a LinearRegression Ridge model for the best estimates
#First convert categorical values to dummy values, and drop SalePrice. Then normalize columns to (0,1)
def scale_minmax(col):
    return (col-col.min())/(col.max()-col.min())
housing_frontage = pd.get_dummies(housing_train)
for col in housing_frontage.drop('LotFrontage', axis=1).columns:
    housing_frontage[col] = scale_minmax(housing_frontage[col])

#Create our X and y values for LotFrontage to use in our Ridge Model
lf_train = housing_frontage.dropna()
lf_train_y = lf_train.LotFrontage
lf_train_X = lf_train.drop('LotFrontage', axis=1)

#Fit and predict our model
lr = Ridge()
lr.fit(lf_train_X, lf_train_y)
lf_pred = lr.predict(lf_train_X)

#Fill our NaN values with our model predictions
nan_frontage = housing_train.LotFrontage.isnull()
X = housing_frontage[nan_frontage].drop('LotFrontage',axis=1)
y = lr.predict(X)
housing_train.loc[nan_frontage,'LotFrontage'] = y

#Remaining NaN values:
cols_with_na = housing_train.isnull().sum()
cols_with_na = cols_with_na[cols_with_na>0]
#print(cols_with_na.sort_values(ascending=False))
rows_with_na = housing_train.isnull().sum(axis=1)
rows_with_na = rows_with_na[rows_with_na>0]
#print(rows_with_na.sort_values(ascending=False))
#Fill remaining nans with mode in that column
for col in cols_with_na.index:
   housing_train[col].fillna(housing_train[col].mode()[0], inplace=True)

#Now no more NaN values apart from SalePrice in test data (Missing SalePrice data is the data to be predicted)
#print(housing_train.info())

#Now lets deal with our different datatypes
cat_cols = [x for x in housing_train.columns if housing_train[x].dtype == 'object']
int_cols = [x for x in housing_train.columns if housing_train[x].dtype == 'int64']
float_cols = [x for x in housing_train.columns if housing_train[x].dtype == 'float64']
cat_cols.append('MSSubClass') #This appears categorical but was put in int_cols for some reason
remove_from_int = ['MSSubClass', 'Id']
int_cols = [x for x in int_cols if x not in remove_from_int]
num_cols = int_cols + float_cols

#Now to encode our categorical data with CategoricalEncoder
def encode_cat(dat):
    cat_encoder = CategoricalEncoder(encoding='onehot-dense')
    dat = dat.astype('str')
    dat_reshaped = dat.values.reshape(-1,1)
    dat_1hot = cat_encoder.fit_transform(dat_reshaped)
    col_names = [dat.name + '_' + str(x) for x in list(cat_encoder.categories_[0])]
    return pd.DataFrame(dat_1hot, columns=col_names)
cat_df = pd.DataFrame()
for x in cat_cols:
    cat_df = pd.concat([cat_df, encode_cat(housing_train[x])], axis=1)
cat_df.index = housing_train.index 
full_df = pd.concat([housing_train[num_cols], cat_df], axis=1)

#Now lets scale and preprocess our data for ML
ss = StandardScaler()
scaled_data = ss.fit_transform(full_df.values)

train_processed = scaled_data[:len(train),:]
test_processed = scaled_data[1460:,:]
log_sp = np.log1p(train['SalePrice'].values).ravel()

#Now to fit our model(Lets use GradientBoostingRegressor)

gbr_reg = GradientBoostingRegressor()
gbr_reg.fit(y=log_sp, X=train_processed)
gbr_pred = np.expm1(gbr_reg.predict(test_processed))

#Lets K-Fold cross validate
scores = cross_val_score(gbr_reg, train_processed, log_sp, scoring='neg_mean_squared_error', cv=10)
gbr_scores = np.sqrt(-scores)
def display_scores(scores):
    print('Scores:', scores)
    print('Mean', scores.mean())
    print('Standard deviation', scores.std())
print(display_scores(gbr_scores))

#Now lets use GridSearchCV to tune our models hyperparameters
param_grid = [
    {'n_estimators': [3,10,20,30,40,50], 'max_depth': [1,2,3,4,5,6,7,8]}
]
grid_search = GridSearchCV(gbr_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(train_processed, log_sp)
# print(grid_search.best_params_)
# print(grid_search.best_estimator_)

#Lets use RandomForestRegression(grid_search.best_estimator_) and test on our test set
final_model = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=5, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=50, n_iter_no_change=None, presort='auto',
             random_state=None, subsample=1.0, tol=0.0001,
             validation_fraction=0.1, verbose=0, warm_start=False)

final_model.fit(y=log_sp, X=train_processed)
final_predictions = np.expm1(gbr_reg.predict(test_processed))
print(final_predictions)

#my_submission4 = pd.DataFrame({'Id': housing_test.Id, 'SalePrice': final_predictions})
# print(my_submission4.head())

#my_submission4.to_csv('submission4.csv', index=False)



