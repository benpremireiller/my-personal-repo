import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Extracting and cleaning data #######################################
data = pd.read_excel(".\\Data\\final_housing_data.xlsx")
data = clean_data(data)
data = pre_process(data)
data.reset_index(drop=True, inplace=True)

# Exploratory analysis  & pre-process #################################
numeric_predictors = ['parking_other', 'garage_spaces', 'lot_size', 'taxes', 'subway_dist', 'mean_room_size', 
                      'n_floors', 'adjusted_n_rooms', 'median_income', 'lot_front', 'lot_depth', 'median_age',
                      'owner_percentage', 'pop_density', 'days_since_covid', 'sqft_estimated', 'sqft_range', 
                      'baths', 'sqft_calc', 'age', 'diff_room', 'n_rooms', 'region_area', 'rental_taxes_per_unit',
                      'rental_taxes_per_bed', 'rental_beds_per_unit', 'custom_built_taxes', 'property_value_taxes']

categorical_predictors = ['district', 'subway_name', 'type', 'driveway', 'garage',  'laundry_level', 
                          'AC_type', 'heating_fuel', 'levels']

binary_predictors = ['fireplace', 'central_vac', 'pool', 'renovated', 'rental_property', 'property_value', 
                     'ravine_lot', 'custom_built'] + exterior_predictors + basement_predictors + room_predictors


#Missing data for all variables
data.apply(lambda x: x.isna().sum())

##Check missing data
#Check for mean difference in houses with na sqft and without
sqft_condition = data.sqft_range.isna()
with_sqft = data[sqft_condition].sold_price
without_sqft = data[~sqft_condition].sold_price

#plot density then do t-test
np.log(with_sqft).plot(kind = 'density', label = 'with_sqft')
np.log(without_sqft).plot(kind = 'density', label = 'without_sqft')
plt.legend()
ttest_ind(np.array(with_sqft), np.array(without_sqft)) #Likely related to predictor

#Check for taxes
tax_condition = data.taxes.isna()
with_tax = data[tax_condition].sold_price
without_tax = data[~tax_condition].sold_price

#plot density
np.log(with_tax).plot(kind = 'density', label = 'with_tax')
np.log(without_tax).plot(kind = 'density', label = 'without_tax')
plt.legend()
ttest_ind(np.array(with_tax), np.array(without_tax)) #Also likely related to predictor

#Check for subway because locator could not find certain addresses
subway_condition = data.taxes.isna()
with_subway = data[subway_condition].sold_price
without_subway = data[~subway_condition].sold_price

#plot density
np.log(with_subway).plot(kind = 'density', label = 'with_subway')
np.log(without_subway).plot(kind = 'density', label = 'without_subway')
plt.legend()
ttest_ind(np.array(with_subway), np.array(without_subway)) #Also, also likely related to predictor
#Maybe for subway_dist use rf imputation method

#Plot outcome vs predictors
sns.pairplot(data=data,
                  x_vars=numeric_columns,
                  y_vars='sold_price',
                  kind = 'reg')

#Plot pair grid
sns.pairplot(data, vars = numeric_columns)
data[data.lot_size < 15000].plot.scatter('sold_price', 'lot_size')

data.boxplot(by = 'garage_spaces', column = 'sold_price')

#%%
test = data[data.rental_beds_per_unit != 0][['MLS_num', 'sold_price', 'beds', 'baths', 'beds_other', 'description', 'sqft_estimated', 'mean_room_size', 'taxes', 'district', 'rental_beds_per_unit', 'rental_bpu_taxes']]
test['taxes_per_unit'] = test.taxes/test.baths
test['taxes_per_bed'] = test.taxes/(test.beds + (test.beds_other /2))

test.loc[test.rental_beds_per_unit <= 1, 'rental_beds_per_unit'] = 1

test['rental_bpu_taxes'] = test.taxes * test.rental_beds_per_unit
#%%
plt.scatter(test.taxes_per_unit, test.sold_price)

#%%
test_data = data[data.rental_beds_per_unit != 0]
test_data = test_data[test_data.rental_units_taxes < 100000]
plt.scatter(test_data.taxes, test_data.sold_price)


#%%
#Find outliers in taxes
g = sns.FacetGrid(data, hue = 'custom_built')
g.map(plt.scatter, 'taxes', 'sold_price', marker="o").add_legend()
plt.tight_layout(rect = (0,0,1.5,1))

# %%
