#%% Import libraries and load data
import numpy as np
import pandas as pd
import datetime
import sys
sys.path.append('\\.')
from ML_functions import find_permutations

data = pd.read_excel("Data\\joined_housing_data.xlsx")

#%% REMOVE OUTLIERS AND DROP DUPLICATES AND USELESS VARIABLES
data = data[data.sold_price < 0.65e7]
data = data.drop_duplicates()
data.drop(['hh_size', 'college_educated'], axis = 1, inplace = True) #Only NAs
data.reset_index(inplace = True, drop = True)

#%% REPLACE CATEGORICAL VARIABLES WITH ORDINAL ONES

#Assign order to sqft series
data.sqft_range = data.sqft_range.replace({'N/A sq. ft.': np.nan, '700–1100 sq. ft.': 900, '1100–1500 sq. ft.': 1300, '1500–2000 sq. ft.': 1750, 
                                            '2000–2500 sq. ft.': 2250, '2500–3000 sq. ft.': 2750, '3000–3500 sq. ft.': 3250, '3500–5000 sq. ft.': 4250})

#Assign order to age
data.age = data.age.replace({'New years': 1, '0-5 years': 2.5, '6-15 years': 10.5, '16-30 years': 23, '31-50 years': 40.5, '51-99 years': 75, '100+ years': 100})

#%% EXTRACT LOT FRONT, DEPTH AND SIZE

#Calculate lot size and convert metres in lot_size to feet
sqm_to_sqft = 10.7639
data.loc[data.lot_size.isnull(), 'lot_size'] = '0 x 0'

contains_metre = data.lot_size.str.contains('metres')
contains_acres = data.lot_size.str.contains('acres')
contains_feet = ~((contains_metre) | (contains_acres))

acre_series = data.lot_size[contains_acres].str.replace('acres', '').str.replace('x', '*').replace('', 0)
metre_series = data.lot_size[contains_metre].str.replace('metres', '').str.replace('x', '*').replace('', 0)
feet_series = data.lot_size[contains_feet].str.replace('feet', '').str.replace('x', '*').replace('', 0)

#Get frontage and depth
data.loc[contains_metre, 'lot_front'] = metre_series.str.split('*').apply(lambda s: s[0].strip())
data.loc[contains_feet, 'lot_front'] = feet_series.str.split('*').apply(lambda s: s[0].strip())
data.loc[contains_acres, 'lot_front'] = acre_series.str.split('*').apply(lambda s: s[0].strip())
data.loc[data.lot_front == '0', 'lot_front'] = np.nan

data.loc[contains_metre, 'lot_depth'] = metre_series.str.split('*').apply(lambda s: s[1].strip())
data.loc[contains_feet, 'lot_depth'] = feet_series.str.split('*').apply(lambda s: s[1].strip())
data.loc[contains_acres, 'lot_depth'] = acre_series.str.split('*').apply(lambda s: s[1].strip())
data.loc[data.lot_depth == '0', 'lot_depth'] = np.nan

#Convert dtypes
data.lot_front = data.lot_front.astype(float)
data.lot_depth = data.lot_depth.astype(float)

#Calculate lot size
data['lot_size'] = data.lot_front * data.lot_depth
data.loc[data.lot_size == 0, 'lot_size'] = np.nan

#%% OTHER PREPROCESSING BEFORE DATA TYPE CONVERSIONS

#Split beds into normal beds and other beds
data['beds_other'] = data.beds.str.split('+').apply(lambda x: x[1] if len(x) > 1 else 0)
data['beds'] = data.beds.str.split('+').apply(lambda x: x[0])

#Fix houses with 0 rooms or 0 sqft calc to be nan
data.loc[data.n_rooms == 0, 'n_rooms'] = np.nan
data.loc[data.sqft_calc == 0, 'sqft_calc'] = np.nan

#Null garage spaces means 0
data.loc[data.garage_spaces.isnull(), 'garage_spaces'] = '0'

#Split the owner and renter percentages
data['owner_renter_divide'] = data.owner_renter_divide.fillna('101/101')
data['owner_percentage'] = data.owner_renter_divide.str.replace('%', '').str.split('/').apply(lambda s: s[0].strip())
data['renter_percentage'] = data.owner_renter_divide.str.replace('%', '').str.split('/').apply(lambda s: s[1].strip())
data.loc[data.owner_percentage == '101', ['renter_percentage', 'owner_percentage']] = np.nan

#%% CHANGE DATA TYPES AND SIMPLIFY VARIABLES

data.subway_dist = data.subway_dist.astype(float).apply(lambda x: round(x, 2))
data.lot_size = data.lot_size.astype(float)
data.list_price = data.list_price.astype(float)
data.sold_price = data.sold_price.astype(float)
data.beds = data.beds.astype(int)
data.beds_other = data.beds_other.astype(int)
data.baths = data.baths.astype(int)
data.parking = data.parking.replace('no', '0').astype(int)
data.date_scraped = pd.to_datetime(data.date_scraped)
data.sqft_calc = data.sqft_calc.astype(float)
data.garage_spaces = data.garage_spaces.astype(float).astype(int)
data.taxes = data.taxes.str.replace('[^0-9]', '', regex = True).astype(float)
data.median_income = data.median_income.str.replace('[^0-9]', '', regex = True).astype(float)
data.total_owners = data.total_owners.str.replace('[^0-9]', '', regex = True).astype(float)
data.total_renters = data.total_renters.str.replace('[^0-9]', '', regex = True).astype(float)
data.total_pop = data.total_pop.str.replace('[^0-9]', '', regex = True).astype(float)
data.pop_density = data.pop_density.str.replace('[^0-9]', '', regex = True).astype(float)
data.sold_date = pd.to_datetime(data.sold_date.str.replace('SOLD', '').str.strip())
data.owner_percentage = data.owner_percentage.astype(float)/100
data.renter_percentage = data.renter_percentage.astype(float)/100
data.description = data.description.str.lower()
data.extras = data.extras.str.lower()
data.room_names = data.room_names.str.lower()
data.basement_details = data.basement_details.str.lower()
data.exterior_type = data.exterior_type.str.lower()
data.room_floors = data.room_floors.str.lower()

#%% OTHER PREPROCESSING AFTER DATA TYPE CONVERSION
 
#Replace sold dates with approx: data from zoocasa is only shown for past 6 months & we only have scraped date, so take the mid point (3 months: 91 days)
data.loc[data.sold_date.isna(), 'sold_date'] = data.loc[data.sold_date.isna(), 'date_scraped'] - datetime.timedelta(days = 91)

#Days since covid first case in Canada (Jan 21)
data['days_since_covid'] = (data.sold_date - pd.to_datetime(datetime.date(year = 2020, month = 1, day = 25))).dt.days

#Total beds
data['beds_total'] = data.beds + data.beds_other

#%%FIX DATA INPUT ERRORS

#Fix bed values errors and calc total beds
data.loc[data.MLS_num.isin(['C4655068']), 'beds'] = 14
data.loc[data.MLS_num.isin([' W4687581']), 'beds_other'] = 1

#Remove errors from bath (from checking housing website)
data.loc[data.MLS_num == 'E4712532', 'baths'] = data.baths.median()

#Fix lot sizes errors by manually looking up addresses (NOTE: biased towards higher lot_size homes)
data.loc[data.MLS_num.isin(['E4748551', 'E4672211', 'E4641267', 'E4711176']), 'lot_size'] = np.nan
data.loc[data.MLS_num.isin(['C4653306']), 'lot_size'] = 6650

#Fix tax values
data.loc[data.MLS_num.isin(['E4728326', 'C4634060']), 'taxes'] = [3600, np.nan]
data.loc[data.taxes <= 100 , 'taxes'] = np.nan
data.loc[data.taxes >= 100000 , 'taxes'] = np.nan

#Remove sqft calc errors: using absolute dev because more robust with the massive values of sqft_calc (example 144000)
data.loc[data.sqft_calc > data.sqft_calc.mean() + (3 * data.sqft_calc.mad()), 'sqft_calc'] = np.nan #NOTE: kinda sketchy

#Fix errors in garage spaces
data.loc[data.MLS_num == 'W4712292', 'garage_spaces'] = 0

#%% CLEAN LOCATION VARIABLES

#Replace errors in district
data.district = data.district.replace({'Roncesvalles':'Parkdale—High Park', 'Kensington Market': 'Spadina—Fort York', 
                                        'Morningside': 'Scarborough—Guildwood', 'The Junction': 'Parkdale—High Park',
                                        'Swansea': 'Parkdale—High Park'})

#Create placeholders for invalid districts that can't be mapped directly because next cleaning steps maps neighborhoods that are 'in' district to district (we need to only have valid district values)
data.district = data.district.replace({'Scarborough': 'Other1', 'East York': 'Other2', 'York': 'Other3','Toronto': 'Other4',
                                       'Old Toronto': 'Other5', 'Etobicoke': 'Other6', 'North York': 'Other7'})

#Map the correct neighborhoods or areas to district if they exist in district
data.loc[(data.neighborhood.isin(data.district)), 'district'] = data.neighborhood[data.neighborhood.isin(data.district)]
data.loc[(data.area.isin(data.district)), 'district'] = data.area[data.area.isin(data.district)]

#Map as many districts from North York as possible because there are no north york districts in district variable
data.loc[data.neighborhood.isin(['Don Valley North', 'Don Mills', 'Parkway East', 'Hillcrest Village', 'Pleasant View', 'Bayview Village']), 'district'] = 'Don Valley North'
data.loc[data.neighborhood.isin(['Willowdale', 'Lansing']), 'district'] = 'North York—Willowdale'
data.loc[data.neighborhood.isin(['York Centre', 'Northwood Park', 'Downsview', 'Wilson Heights', 'William Baker', 'York Heights']), 'district'] = 'York Centre'
data.loc[data.neighborhood.isin(['York Mills', 'Silver Hills']), 'district'] = 'Don Valley West'
data.loc[data.neighborhood.isin(['Bedford Park', 'Lawrence Manor','Lytton Park',  'Glen Park', 'Castlefield Design District']), 'district'] = 'Eglinton—Lawrence'
data.loc[data.neighborhood.isin(['Humberlea', 'Jane & Finch', 'Emery', 'Humber Summit']), 'district'] = 'Humber River—Black Creek'
data.loc[data.neighborhood.isin(['Victoria Village']), 'district'] = 'Don Valley East'
#TODO Still more to map

#Fix other errors


#Replace remaining 'Other' categories with correct values TODO: replace this
data.district = data.district.replace({'Other1': 'Scarborough—Agincourt', 'Other2': 'Toronto—Danforth'})

#For each 'Other#' find the most common value for district for the respective neighborhoood that isn't 'Other#'
for district in ['Other2', 'Other3', 'Other4', 'Other5', 'Other6', 'Other7']:
    other_df = data[data.district == district]
    missing = 0
    for neighbor in other_df.neighborhood.values:
        neighbor_district = data.district[data.neighborhood == neighbor]
        most_common_district = neighbor_district[neighbor_district != district].mode()
        condition = (data.district == district) & (data.neighborhood == neighbor)
        if not most_common_district.empty:
            data.loc[condition, 'district'] = most_common_district
        else:
            missing += 1
            data.loc[condition, 'district'] = 'None'
    #print('Missing district count for', district, ':', missing)
del(other_df)

data.district = data.district.fillna('None')

data = data[data.district != 'None']
data.reset_index(inplace = True, drop = True)

#%% REMAP CATEGRORICAL VARIABLES
#Remap levels
data.levels = data.levels.replace(['2-Storey', 'Bungalow', '3-Storey', 'Bungalow-Raised',
                                  'Backsplit 4', 'Sidesplit 3', 'Backsplit 5', '1 1/2 Storey',
                                  '2 1/2 Storey', 'Sidesplit 4', 'Backsplit 3', 'Bungaloft',
                                  'Sidesplit 5', 'Other', 'Sidesplt-All'], 
                                  ['2-storey','bungalow','3-storey','bungalow',
                                  'split 4', 'split 3','split 5', '1.5-storey', '2.5-storey','split 4',
                                  'split 3', 'bungalow', 'split 5', 'other', 'other']) 

#Remove sparse categorical values and fix values
data.heating_fuel = data.heating_fuel.replace({'Grnd Srce': 'Other', 
                                                'Propane': 'Other', 
                                                'Wood': 'Other'})

data.type = data.type.replace({'Triplex': 'Multiplex', 
                                'Fourplex': 'Multiplex'})

#%% EXTRACT AND ENGINEER NEW FEATURES

#Extract as many sqft values as possible (only ~250)
ungrouped_sqft = data.description.str.extractall('(?<!lot )([0-9,]+) sq(?:uare)?\.? ?f(?:oo|ee)?t\.?(?! lot)').reset_index()
ungrouped_sqft['sqft'] = ungrouped_sqft[0].str.replace(',','').astype(float)
sqft_series = ungrouped_sqft.groupby(['level_0'])['sqft'].sum().reindex(range(0, len(data)))

priority_first = lambda s1, s2: [first if np.isnan(second) else second for first, second in zip(s1, s2)]

data['sqft_merged'] = priority_first(data.sqft_range, sqft_series)
data.loc[data.MLS_num.isin(['W4680026','C4641406', #Fix a few that were actually lot sizes or errors
                            'C4731304','W4653158']), 'sqft_merged'] = [6600, np.nan, 6000, np.nan]

#Average room size
data['mean_room_size'] = data.sqft_calc / data.n_rooms

#Region size in square miles
data['region_area'] = data.total_pop / data.pop_density

#Other parking
data['parking_other'] = (data.parking - data.garage_spaces).apply(lambda x: 0 if x < 0 else x)

#Flag for renovated in description 
data['renovated'] = data.description.str.contains('renovated') * 1

#Flag for pool
data['pool'] = ((data.description.str.contains('pool(?! table)(?!-size)(?! size)(?!s)')) & 
                ~(data.description.str.contains('(close|walk|steps|near|drive|mins|minute|seconds)[^\\.\\!]+pool'))) * 1

#Flag for hot tub
data['hot_tub'] = data.description.str.contains('hot[- ]?tub') * 1

#Adjusted n_rooms
basement_rooms = (~data.basement_details.str.contains('none') * 1) + (data.basement_details.str.contains('and') * 1) #NOTE fix
bed_bath = data.room_names.str.count('bedroom|master|bathroom')
other_rooms = data.n_rooms - bed_bath

data['adjusted_n_rooms'] = other_rooms + data.beds_total + data.baths + basement_rooms
data.loc[data.adjusted_n_rooms.isna(), 'adjusted_n_rooms'] = other_rooms.median() + data.beds_total + data.baths + basement_rooms

#Estimate sqftage of house
data['sqft_estimated'] = data.mean_room_size * data.adjusted_n_rooms
data[['sqft_estimated', 'sqft_range']].corr() #Check: 0.82 corr

#Difference between n_rooms and average n_rooms for district
data['diff_room'] = data.adjusted_n_rooms - data.groupby('district')['adjusted_n_rooms'].transform('mean')


#%% HOUSES SOLD FOR PURPOSES OTHER THAN HOMES 

#Flag for house being sold for lot value
data['property_value'] = data.description.str.contains("(builder(?!'s))|(blder(?!'s))|(buildable)|(build your)|(sold as is)") * 1
data.loc[data.MLS_num.isin(['W4745715']), 'property_value'] = 1
data.loc[data.MLS_num.isin(['C4677376', 'W4652577']), 'property_value'] = 0

#Create flag for houses that are used for rental purposes and manually fix errors
data['rental_property'] = (data.description.str.contains('(?<!ac-)(?<!a/c-)(?<!ac )(?<!a/c )(?<!a.c )(?<!a-c )(?<!opport)(?<!comm)(?<!wall)(?<!wall )(?<!end-)(?<!inside-)(?<!end )(?<!inside )unit(?!ed)|rented')) * 1
data.loc[data.MLS_num.isin(['C4655068', 'W4687581', 'W4646585']), 'rental_property'] = 1
data.loc[data.MLS_num.isin(['C4735576', 'C4703446', 'W4661973', 'C4644839']), 'rental_property'] = 0

#Taxes per rental units assuming 1 bath per unit
data['rental_taxes_per_unit'] = data.rental_property * (data.taxes / data.baths)

#Taxes per bed
data['rental_taxes_per_bed'] = data.rental_property * (data.taxes/ (data.beds + (data.beds_other/2))) #Treat 'other' beds as half beds

#Number of beds per unit using same assumption as above
data['rental_beds_per_unit'] = data.rental_property * ((data.beds + (data.beds_other/2))/data.baths)

#Custom built homes (generally higher priced)
data['custom_built'] = data.description.str.contains('(custom (built )?.{,25}(home|house))|(professionally designed)') * 1

#Interaction terms
data['custom_built_taxes'] = data.custom_built * data.taxes 
data['property_value_taxes'] = data.property_value * data.taxes

#%% OTHER VARIABLES

#Number of floors
data.loc[data.room_floors.isna(), 'room_floors'] = 'none'
has_inbetwn = data['room_floors'].str.contains('in betwn')

floors = data.room_floors.str.replace(',', '').str.replace('sub-bsmt', '').str.replace('bsmt', '').str.replace('ground', 'main').str.replace('upper', '2nd')
floors = floors.str.replace('in betwn', '').str.replace('lower', 'main').str.replace(' {2,}', ' ').str.strip().str.split(' ')

data['n_floors'] = floors.apply(lambda x: len(set(x)) if x[0] != 'none' else 0)
data.loc[data.room_floors == 'none', 'n_floors'] = data.n_floors.median() 
data['n_floors'] = data['n_floors'] + (has_inbetwn * 0.25) #treat inbtween floors as a quarter of floor

#Average rooms per floor
data['mean_room_floors'] = data['adjusted_n_rooms'] / data['n_floors']

#Flag for properties with large lots in the back
data['ravine_lot'] = data.description.str.contains('(ravine[- ]?lot)|(ravine home)') * 1

#Replace sparse heat type valules
data.heat_type = data.heat_type.replace({'Fan Coil': 'Other'})

#Replace sparse heat fuel values
data.heating_fuel = data.heating_fuel.replace({'Wood': 'Other', 'Grnd Srce': 'Other'})

#Create several flags for included appliances (doesn't really work: severeal houses not showing appliances in extras text)
laundry = data.laundry_level.notna() * 1
stove = data.extras.str.contains('stove') * 1
oven = data.extras.str.contains('oven') * 1
fridge = data.extras.str.contains('fridge') * 1
washer = data.extras.str.contains('(?<!dish)(?<!dish )(?<!dish-)(washer)|( dw)') * 1
dryer = data.extras.str.contains('dryer')
dwasher = data.extras.str.contains('dish[- ]?washer') * 1

data['appliance_ratio'] = (laundry + stove + oven + fridge + washer + dryer + dwasher) / 7
data.loc[data.extras.isna(), 'appliance_ratio'] = data.appliance_ratio.mean()

#Create flags for certain room types
data.room_names = data.room_names.fillna('other')
data['room_rec'] = data.room_names.str.contains('rec') * 1
data['room_den'] = data.room_names.str.contains('den') * 1
data['room_media'] = data.room_names.str.contains('media/ent') * 1
data['room_exercise'] = data.room_names.str.contains('exercise') * 1
data['room_mud'] = data.room_names.str.contains('mudroom') * 1
data['room_game'] = data.room_names.str.contains('games') * 1
data['room_breakfast'] = data.room_names.str.contains('breakfast') * 1
data['room_theatre'] = ((data.description.str.contains('theatre')) & ~(data.description.str.contains('(close|walk|steps|near|drive|mins|minute|seconds)[^\\.\\!]+theatre'))) * 1

room_predictors = list(data.columns[data.columns.str.contains('^room_')].drop(['room_names', 'room_floors']))

#Extract features from basement details 
data.basement_details = data.basement_details.fillna('other')

basement_values = data.basement_details.value_counts().index

data['basement_none'] = data.basement_details.str.contains('none') * 1
data['basement_partfin'] = data.basement_details.str.contains('part fin') * 1
data['basement_fin'] = data.basement_details.str.contains('(?<!part )(?<!un)fin') * 1
data['basement_unfin'] = data.basement_details.str.contains('unfin') * 1
data['basement_wo'] = data.basement_details.str.contains('w/o') * 1
data['basement_apartment'] = data.basement_details.str.contains('apartment') * 1
data['basement_full'] = data.basement_details.str.contains('full') * 1
data['basement_sepentrance'] = data.basement_details.str.contains('sep entrance') * 1
data['basement_crawl'] = data.basement_details.str.contains('crawl space') * 1
data['basement_wu'] = data.basement_details.str.contains('walk-up') * 1

basement_predictors = list(data.columns[data.columns.str.contains('^basement_')].drop('basement_details'))

#Extract info from exterior_type
data.exterior_type = data.exterior_type.fillna('other')

exterior_values = data.exterior_type.value_counts().index

data['exterior_brick'] = data.exterior_type.str.contains('brick') * 1
data['exterior_alum'] = data.exterior_type.str.contains('alum') * 1
data['exterior_vinyl'] = data.exterior_type.str.contains('vinyl') * 1
data['exterior_stone'] = data.exterior_type.str.contains('stone') * 1
data['exterior_wood'] = data.exterior_type.str.contains('wood') * 1
data['exterior_metal'] = data.exterior_type.str.contains('metal') * 1
data['exterior_stucco'] = data.exterior_type.str.contains('stucco') * 1

exterior_predictors = list(data.columns[data.columns.str.contains('^exterior_')].drop('exterior_type'))

#Binarize other binary variables
data.fireplace = (data.fireplace == 'Yes') * 1
data.central_vac = (data.central_vac == 'Yes') * 1


#Define columns and transform
numeric_predictors = ['parking_other', 'garage_spaces', 'lot_size', 'taxes', 'subway_dist', 'mean_room_size', 
                      'n_floors', 'adjusted_n_rooms', 'median_income', 'lot_front', 'lot_depth', 'median_age',
                      'owner_percentage', 'pop_density', 'days_since_covid', 'sqft_estimated', 'sqft_range', 
                      'baths', 'sqft_calc', 'age', 'diff_room', 'n_rooms', 'region_area', 'rental_taxes_per_unit',
                      'rental_taxes_per_bed', 'rental_beds_per_unit', 'custom_built_taxes', 'property_value_taxes']

categorical_predictors = ['district', 'subway_name', 'type', 'driveway', 'garage',  'laundry_level', 
                          'AC_type', 'heating_fuel', 'levels']

binary_predictors = ['fireplace', 'central_vac', 'pool', 'renovated', 'rental_property', 'property_value', 
                     'ravine_lot', 'custom_built'] + exterior_predictors + basement_predictors + room_predictors

data[binary_predictors] = data[binary_predictors].astype(int)

#Delete placeholder variables crowding variables viewer
del(contains_acres, contains_feet, contains_metre, metre_series, acre_series, feet_series, sqm_to_sqft)
del(washer, dryer, laundry, stove, oven, dwasher, fridge)
del(has_inbetwn, floors)
del(neighbor, neighbor_district, most_common_district, district, condition, missing)
del(basement_rooms, bed_bath, other_rooms, exterior_values, basement_values)
del(exterior_predictors, basement_predictors, room_predictors)
del(ungrouped_sqft, sqft_series)


#data['room_other'] = data[room_predictors].apply(lambda x: (x == 0).all(), axis = 1) * 1

#%%
###Get second order and interactions of top n
n = 8
higher_order_predictors = data.drop('list_price', axis = 1).corr()['sold_price'].sort_values().tail(n + 1).drop('sold_price').index

accumulated_predictors = []
for predictor in higher_order_predictors:
    #Squared terms
    data[predictor + '_^2'] = np.square(data[predictor])
    numeric_predictors.append(predictor + '_^2')
    accumulated_predictors.append(predictor)
    
    for other_predictor in higher_order_predictors.drop(accumulated_predictors):
        #Interaction terms
        data[predictor + '_' + other_predictor] = data[predictor] * data[other_predictor]
        numeric_predictors.append(predictor + '_' + other_predictor)

# %%
