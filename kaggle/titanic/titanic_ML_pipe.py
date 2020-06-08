import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, RepeatedKFold, KFold, train_test_split, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import re
import xgboost as xgb
from xgboost import XGBClassifier
#from pandas_ml import ConfusionMatrix


#Preprocessing steps
numeric_columns = ['family_size', 'Pclass', 'Age', 'Fare', 'cabin_deck']
onehot_columns = ['title', 'Embarked']
binary_columns = ['Sex', 'boarded_with_children']


X = train_data.drop(['Survived'], axis = 1)
Y = train_data[['Survived']]

#Define training models
classifiers = [LogisticRegression(),
               SVC(),
               KNeighborsClassifier(),
               RandomForestClassifier(),
               GradientBoostingClassifier(),
               MLPClassifier(),
               AdaBoostClassifier(),
               DecisionTreeClassifier()]

#If more than 3 params, order by facet aspect, color, then X-axis
#If less than 3 params, order by aspect then X-axis
train_hyperparameters = [{
    'mod__penalty': ['none', 'l2'],
    'mod__solver': ['saga'],
    'mod__C': [2**a for a in range(-1, 5)]
}, {
    'mod__gamma': [0.01, 0.025, 0.05],
    'mod__degree': [1, 2],
    'mod__C': [2**a for a in range(-1, 5)],
    'mod__kernel': ['poly']
}, {
    'mod__weights': ['uniform', 'distance'],
    'mod__algorithm': ['ball_tree', 'kd_tree', 'brute'],
    'mod__n_neighbors': [2**a for a in range(1, 6)]
}, {
    'mod__max_depth': [1,3,5,7],
    'mod__n_estimators': [10, 50, 100, 150, 200]
},  {
    'mod__learning_rate': [0.01, 0.05],
    'mod__max_depth': [1,3,5],
    'mod__n_estimators': [10, 50, 100, 150, 200]
}, {
    'mod__solver': ['adam', 'lbfgs'],
    'mod__activation': ['relu', 'logistic'],
    'mod__max_iter': [10, 50, 100, 150, 200],
    'mod__hidden_layer_sizes': [(25,)]
}, {
    'mod__learning_rate': [0.5, 1, 1.5],
    'mod__n_estimators': [10, 50, 100, 150, 200]
}, {
    'mod__splitter': ['best', 'random'],
    'mod__max_depth': [1,3,5,7]
}]

final_hyperparameters = [{
    'mod__penalty': ['l2'],
    'mod__solver': ['saga'],
    'mod__C': [2]
}, {
    'mod__gamma': [0.025],
    'mod__degree': [2],
    'mod__C': [4],
    'mod__kernel': ['rbf']
}, {
    'mod__weights': ['uniform'],
    'mod__algorithm': ['brute'],
    'mod__n_neighbors': [5]
}, {
    'mod__max_depth': [7],
    'mod__n_estimators': [50]
},  {
    'mod__learning_rate': [0.01],
    'mod__max_depth': [3],
    'mod__n_estimators': [100]
}, {
    'mod__solver': ['adam'],
    'mod__activation': ['relu'],
    'mod__max_iter': [100],
    'mod__hidden_layer_sizes': [(16,)]
}, {
    'mod__learning_rate': [0.3],
    'mod__n_estimators': [50]
}, {
    'mod__splitter': ['random'],
    'mod__max_depth': [5]
}]

#Split data
#X_train,  X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.05, stratify = Y)

cv_predictions = []
classification_output = []

#Train each model

for params, model in zip(train_hyperparameters, classifiers):

    #Model name
    name = re.findall('(\w+)\\(', str(model))[0]

    #Transform and pipeline
    onehot_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'None')),
    ('encode', OneHotEncoder(handle_unknown='ignore'))])
 
    numeric_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(KNeighborsRegressor(n_neighbors=3))),
        ('scaler', StandardScaler())])
    

    preprocess = ColumnTransformer(transformers = [('onehot', onehot_transformer, onehot_columns),
                                                   ('num', numeric_transformer, numeric_columns),
                                                   ('binary', OrdinalEncoder(), binary_columns)])

    pipeline = Pipeline(steps = [('preprocess', preprocess), ('mod', model)])

    #Fit the current model
    cv = RepeatedKFold(n_splits=5, n_repeats = 10)

    full_pipeline = GridSearchCV(pipeline, param_grid = params, cv = cv,
                        n_jobs = -1, verbose = 1)

    print('Fitting', name, 'model **************************************************')
    full_pipeline.fit(X, Y)

    best_params = full_pipeline.best_params_
    print('Best estimator params: ', best_params)
    
    #Extract parameters for cross val predict because it needs a certain format
    cv_params = {key.split("__")[1]: value for key, value in best_params.items()}

    #Get best cross val predictions
    cv_pipeline = Pipeline(steps = [('preprocess', preprocess), ('mod', model.set_params(**cv_params))])
    cv_prediction = cross_val_predict(cv_pipeline, X, Y, cv = 5, n_jobs = -1)
    cv_predictions.append(cv_prediction)

    #Get crossval results
    cv_data = pd.DataFrame(full_pipeline.cv_results_)

    #Add param_ to paramter keys which so we can reference dataframe names of cv_data
    keys = [*map(lambda x: 'param_' + x, list(params.keys()))]

    #Plot cv results over different hyperparameters

    sns.set(style="ticks")
    if len(keys) > 2:
        g = sns.FacetGrid(cv_data, col=keys[0],  hue = keys[1])
        g.map(plt.plot, keys[2], "mean_test_score", marker="o").add_legend()
        plt.subplots_adjust(top=0.8)
        g.fig.suptitle(name)
        plt.show()

    elif len(keys) == 2:    
        g = sns.FacetGrid(cv_data, col=keys[0])
        g.map(plt.plot, keys[1], "mean_test_score", marker="o")
        plt.subplots_adjust(top=0.8)
        g.fig.suptitle(name)
        plt.show()

    ###Final evaluation parameters on train and test sets
    #Predicted
    Y_hat_test = full_pipeline.predict(X).flatten()
    Y_hat_train = full_pipeline.predict(X).flatten()
    #Actual
    Y_vals= np.array(Y).flatten()

    #Accuracy measures TODO: add ROC
    best_model = cv_data.rank_test_score == 1
    val_acc = cv_data.loc[best_model, 'mean_test_score'].values[0]
    val_std = cv_data.loc[best_model, 'std_test_score'].values[0]
    train_acc = accuracy_score(Y_vals, Y_hat_train)

    ###Confusion matrix plot
    print('Printing confusion matrix of train data for: ', name)
    print(confusion_matrix(Y_vals, Y_hat_test))
    print('Printing confusion matrix of val data for: ', name)
    print(confusion_matrix(Y_vals, cv_prediction))

    ###Append scores to list
    classification_output.append([name, best_params, train_acc, val_acc, val_std, full_pipeline])

#Create dataframe of final scores
classification_output = pd.DataFrame(classification_output, columns = ['model_name', 'best_params', 'train_acc', 'val_acc', 'val_std', 'model_object'])
classification_output

#Get first layer predictions
layer_predictions = np.zeros((len(test_data), len(classification_output)))
for i, model in enumerate(classification_output.model_object):
    layer_predictions[:, i] = model.predict(test_data)

#Find correlation between results
predictions_df = pd.DataFrame(layer_predictions)
predictions_df.columns = classification_output.model_name
predictions_df.corr()

#Train the stacking classifier ***************************************************************

#dtrain = xgb.DMatrix(data = np.array(cv_predictions).T, label = Y)
#xgb_reg.train(xgb_params, dtrain, 10)

xgb_params = {
    'n_estimators': [1000],
    'eta': [0.5],
    'max_depth': [1]
}

clf = XGBClassifier()
xgb_grid = GridSearchCV(clf, param_grid = xgb_params, cv = RepeatedKFold(),
                        n_jobs = -1, verbose = 1)
xgb_grid.fit(np.array(cv_predictions).T, Y)
xgb_cv_results = pd.DataFrame(xgb_grid.cv_results_)

#Plot cv results
g = sns.FacetGrid(xgb_cv_results, col='param_eta', hue = 'param_n_estimators')
g.map(plt.plot, 'param_max_depth', "mean_test_score", marker="o").add_legend()

#Fit stack model******************************************************************************

stack_predictions = pd.Series(xgb_grid.predict(layer_predictions))
submission = pd.concat([test_data.PassengerId, stack_predictions], axis = 1, ignore_index = True)
submission.columns = ['PassengerId', 'Survived']
submission.Survived = submission.Survived.astype(int)
submission.to_csv('.\\Data\Titanic\\titanic_submission.csv', index = False)

#Fit indivual models**************************************************************************

model_number = 1

submission = pd.concat([test_data.PassengerId, predictions_df.iloc[:, model_number]], axis = 1, ignore_index = True) #Try names argument for col names
submission.columns = ['PassengerId', 'Survived']
submission.Survived = submission.Survived.astype(int)
submission.to_csv('.\\Data\Titanic\\titanic_submission.csv', index = False)