#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, RepeatedKFold, KFold, train_test_split, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder, PolynomialFeatures, OrdinalEncoder, FunctionTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
import re
import sys
sys.path.append('\\.')
from ML_functions import within_n_std, inverse

#Naive 'mean' model RMSE: 653333, MAE:460000
binary_categories = np.array([[0, 1]] * len(binary_predictors))

#Preprocessing steps
X = data.drop('sold_price', axis = 1)
Y = data[['sold_price']]

#Define models
models = [HuberRegressor(),
          MLPRegressor(),
          SVR(),
          RandomForestRegressor(),
          GradientBoostingRegressor(),
          XGBRegressor(),           
          DecisionTreeRegressor()]

#Order by column, x-axis, color then row
hyperparameters = [{
    'mod__alpha': [0.0001, 0.001, 0.01],
    'mod__epsilon': [1, 1.5, 1.75, 2]
}, {
    'mod__solver': ['adam'],
    'mod__max_iter': [10, 20, 30, 40, 50, 60, 70, 80, 90],
    'mod__hidden_layer_sizes': [(32,), (64,), (72,), (16, 16)]
}, {
    'mod__gamma': [0.01],
    'mod__C': [2**a for a in range(-1, 6)],
    'mod__degree': [2]
}, {
    'mod__criterion': ['mse', 'mae'],
    'mod__n_estimators': [15, 25, 35, 45, 55, 65, 75],
    'mod__max_depth': [5, 7]
}, {
    'mod__loss': ['ls', 'lad', 'huber'],
    'mod__n_estimators': [10, 50, 100, 125, 175, 250],    
    'mod__learning_rate': [0.075],
    'mod__max_depth': [5, 7]
}, {
    'mod__eta': [0.2, 0.4],
    'mod__n_estimators': [10, 25, 50, 75, 90],
    'mod__max_depth': [3, 5, 7]
}, {
    'mod__splitter': ['best', 'random'],
    'mod__max_depth': [3,5,7,9,12,15,18]
}]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 3)

cv_predictions = []
regression_output = []

for params, model in zip(hyperparameters, models):

    #Model name as string
    name = re.findall('(\w+)\\(', str(model))[0]

    #Add prefix to parameters to get df keys for graphs
    keys = [*map(lambda x: 'param_' + x, list(params.keys()))]

    #Create tranformers
    onehot_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'None')),
        ('encode', OneHotEncoder(handle_unknown='ignore'))])

    numeric_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(KNeighborsRegressor())),
        ('scaler', PowerTransformer(method = 'yeo-johnson'))]) 

    binary_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy = 'constant', fill_value = 0)),
    ('encode', OrdinalEncoder(categories = binary_categories))])

    preprocess = ColumnTransformer(transformers = [('num', numeric_transformer, numeric_predictors),
                                                    ('onehot', onehot_transformer, categorical_predictors),
                                                   ('binary', binary_transformer, binary_predictors)])

    #Build model pipeline
    pipeline = Pipeline(steps = [('preprocess', preprocess), ('mod', model)])

    cv = RepeatedKFold(n_splits=5, n_repeats = 5)

    grid = GridSearchCV(pipeline,  
                        param_grid = params, 
                        cv = 7, 
                        refit = lambda cvd: within_n_std(cvd, keys, n = 0.5),
                        n_jobs = -1,
                        verbose = 1, 
                        scoring = 'neg_root_mean_squared_error')

    target_transformer = PowerTransformer(method = 'box-cox')

    full_pipeline = TransformedTargetRegressor(regressor=grid, transformer = target_transformer)

    #Fit the model
    print('\nFitting', name, 'model ********************************************')
    full_pipeline.fit(X_train, Y_train)

    grid_fit = full_pipeline.regressor_
    best_params = grid_fit.best_params_
    print('Refitted estimator params: ', best_params)

    #Extract parameters for cross_val_predict() because it requires a certain format
    cv_params = {key.split("__")[1]: value for key, value in best_params.items()}

    #Get best cross val predictions
    cv_pipeline = Pipeline(steps = [('preprocess', preprocess), ('mod', model.set_params(**cv_params))])
    cv_prediction = cross_val_predict(cv_pipeline, X_train, Y_train, cv = 7, n_jobs = -1)
    cv_predictions.append(cv_prediction.flatten())

    ###Plot eval metric over params
    cv_data = pd.DataFrame(grid_fit.cv_results_)
    cv_data.mean_test_score = -cv_data.mean_test_score

    sns.set(style="ticks")
    if len(keys) >= 4:
        g = sns.FacetGrid(cv_data, col=keys[0], row=keys[3], hue = keys[2])
        g.map(plt.plot, keys[1], "mean_test_score", marker="o").add_legend()
        [plt.setp(ax.texts, text="") for ax in g.axes.flat]
        g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
        plt.subplots_adjust(top=0.8)
        g.fig.suptitle('Validation Scores')
        plt.show()

    if len(keys) == 3:
        g = sns.FacetGrid(cv_data, col=keys[0], hue = keys[2])
        g.map(plt.plot, keys[1], "mean_test_score", marker="o").add_legend()
        [plt.setp(ax.texts, text="") for ax in g.axes.flat]
        g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
        plt.subplots_adjust(top=0.8)
        g.fig.suptitle('Validation Scores')
        plt.show()

    elif len(keys) == 2:    
        g = sns.FacetGrid(cv_data, col=keys[0])
        g.map(plt.plot, keys[1], "mean_test_score", marker="o")
        [plt.setp(ax.texts, text="") for ax in g.axes.flat]
        g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
        plt.subplots_adjust(top=0.8)
        g.fig.suptitle('Validation Scores')
        plt.show()
    plt.close()
 
    ###Final evaluation parameters on train, test and val sets

    #Predicted
    Y_hat_test = full_pipeline.predict(X_test).flatten()
    Y_hat_train = full_pipeline.predict(X_train).flatten()
    #Actual
    Y_vals_test = np.array(Y_test).flatten()
    Y_vals_train = np.array(Y_train).flatten()

    #RMSE
    rmse_test = mean_squared_error(Y_vals_test, Y_hat_test, squared = False)
    rmse_train = mean_squared_error(Y_vals_train, Y_hat_train, squared = False)
    #MAE
    mae_test = mean_absolute_error(Y_vals_test, Y_hat_test)
    mae_train = mean_absolute_error(Y_vals_train, Y_hat_train)
    #R^2
    rsquared_test = r2_score(Y_vals_test, Y_hat_test)
    rsquared_train = r2_score(Y_vals_train, Y_hat_train)

    ###Residual vs actual plot and predicted vs actual plot
    diff_train = Y_vals_train - Y_hat_train
    plt.subplot(1, 2, 1)
    plt.scatter(Y_vals_train, diff_train)
    plt.plot(Y_vals_train, np.repeat(0, len(Y_vals_train)),'k-')
    plt.subplots_adjust(top=0.8)
    plt.title('Residuals vs Actual')

    plt.subplot(1, 2, 2)
    plt.scatter(Y_vals_train, Y_hat_train, color = 'magenta')
    plt.axis('equal')
    plt.plot(Y_vals_train, Y_vals_train,'k-')
    plt.subplots_adjust(top=0.8)
    plt.title('Predicted vs Actual')

    #Adjust size and show
    plt.tight_layout(rect = (0,0,1.5,1))
    plt.show()
    plt.close()

    ###Append scores to list
    regression_output.append([name, best_params, rmse_train, rmse_test, mae_train, mae_test,
                              rsquared_train, rsquared_test, full_pipeline])

regression_output = pd.DataFrame(regression_output, columns = ['model', 'best_params','rmse_train', 'rmse_test', 
                                                               'mae_train', 'mae_test','rsquared_train', 'rsquared_test', 'model_object'])
regression_output

#%%
#Get first layer predictions
layer_predictions = np.zeros((len(X_test), len(regression_output)))
for i, model in enumerate(regression_output.model_object):
    layer_predictions[:, i] = model.predict(X_test).flatten()

#Find correlation between results
predictions_df = pd.DataFrame(layer_predictions)
predictions_df.columns = regression_output.model
predictions_df.corr()

#Create stacking model
X_stack = np.array(cv_predictions).T
X_stack = np.delete(X_stack, 1, axis=1) #MPL regression returning very small values
layer_predictions = np.delete(layer_predictions, 1, axis=1) #Remove MPL predictions also
stack = LinearRegression(fit_intercept = False)
stack.fit(X_stack, Y_train)
stack_predictions = stack.predict(layer_predictions).flatten()

#%%
#Eval model
print('RMSE:', mean_squared_error(Y_test, stack_predictions, squared = False))
print('MAE:',mean_absolute_error(Y_test, stack_predictions))
print('R-squared:', r2_score(Y_test, stack_predictions))

#Lowest score on test set so far [RMSE:187500, MAE:124000, R^2:0.9]


# %%
