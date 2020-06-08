import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from scipy.stats import ttest_ind

train_data = pd.read_csv('.\\Data\\Titanic\\train.csv')
test_data = pd.read_csv('.\\Data\\Titanic\\test.csv')

merged_data = pd.concat([train_data, test_data])

test_ids = test_data.PassengerId

###Feature engineering 

#Fix fare values
#Check identical ticket values
ticket_counts = merged_data.groupby('Ticket')['Ticket'].count() 
duplicate_tickets = ticket_counts[ticket_counts > 1].index

train_copy = train_data

test = merged_data[merged_data.Ticket.isin(duplicate_tickets)].sort_values('Ticket').drop('Name', axis = 1)
merged_data['boarded_with_children'] = 0

#Split cost of ticket over group and check if there are parents and children
for ticket in duplicate_tickets:
    #Split cost of ticket
    ticket_fares = merged_data.Fare[merged_data.Ticket == ticket].values
    cost_per_person = ticket_fares[0] / len(ticket_fares)
    merged_data.loc[merged_data.Ticket == ticket, 'Fare'] = cost_per_person

    #Check if people with same ticket have children
    are_children = (merged_data.Parch[merged_data.Ticket == ticket].values > 1).sum()
    if are_children > 0:
        merged_data.loc[merged_data.Ticket == ticket, 'boarded_with_children'] = 1


#Ticket prefix 
merged_data['ticket_prefix'] = merged_data.Ticket.str.extract('(.+ |[a-zA-Z]+)?(\d*)')[0].str.strip()
merged_data['ticket_prefix'] = merged_data['ticket_prefix'].replace({
       'A. 2.': 'A', 'A./5.': 'A', 'A.5.': 'A', 'A/4': 'A',
       'A/4.': 'A', 'A/5': 'A', 'A/5.': 'A', 'A/S': 'A',
       'A4.': 'A', 'AQ/3.': 'A', 'AQ/4': 'A', 'C.A.': 'C', 
       'C.A./SOTON': 'C', 'CA': 'C', 'CA.': 'C','F.C.': 'FC', 
       'F.C.C.': 'FC', 'Fa': 'FC', 'LP': 'PP', 'P/PP': 'PP', 'PC': 'PP', 
       'S.C./A.4.': 'SC', 'S.C./PARIS': 'SC', 'S.O./P.P.': 'SO', 
       'S.O.C.': 'SO', 'S.O.P.': 'SO', 'S.P.': 'SO', 'S.W./PP': 'PP',
       'SC/A.3': 'SC', 'SC/A4': 'SC', 'SC/AH':'SC', 'SC/AH Basle': 'SC', 
       'SC/PARIS': 'SC', 'SC/Paris': 'SC', 'SCO/W': 'SC', 'SO/C': 'SC', 
       'SOTON/O.Q.': 'SOTON', 'SOTON/O2': 'SOTON', 'SOTON/OQ': 'SOTON',
       'STON/O 2.': 'STON', 'STON/O2.': 'STON', 'STON/OQ.': 'STON', 'SW/PP': np.nan, 
       'W./C.': 'WC', 'W.E.P.': 'WEP', 'W/C': 'WC', 'WE/P': 'WEP'})

merged_data['ticket_number'] = merged_data.Ticket.str.extract('(.+ |[a-zA-Z]+)?(\d*)')[1]
merged_data['ticket_len'] = merged_data.ticket_number.apply(len)
merged_data['ticket_first'] = merged_data['ticket_number'].str.extract('(^\d{1})')[0]
merged_data['ticket_first'] = merged_data['ticket_first'].apply(lambda x: int(x) if x in ['1','2','3'] else 0)

#Get cabin deck and number
merged_data['cabin_deck'] = merged_data.Cabin.str.extract('(^\w)')[0]
merged_data['cabin_deck'] = merged_data['cabin_deck'].replace({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8, np.nan: 0}).astype(int)

merged_data['cabin_raw_number'] = merged_data.Cabin.str.extract('(^\w)(\d+$)')[1]
merged_data['cabin_raw_number'] = merged_data['cabin_raw_number'].astype(float)

#Section the ship
merged_data['ship_area'] = np.where(merged_data.cabin_raw_number < 40, 1,
                            np.where(merged_data.cabin_raw_number < 80, 2, 
                            np.where(merged_data.cabin_raw_number < 200, 3, 0)))

merged_data['ship_area'].astype(int)

#Total family size
merged_data['family_size'] = merged_data['SibSp'] + merged_data['Parch']

#Check whether passenger is alone
merged_data['is_alone'] = (merged_data['family_size'] == 0) * 1

#Get titles
merged_data['title'] = merged_data.Name.str.extract('(^.+\\,) (\w+\.)')[1]
merged_data['title'] = merged_data['title'].replace({'Jonkheer.': 'Other', 'Sir.': 'Other', 'Capt.': 'Military', 'Mme.': 'Other', 'Don.' :'Other', 'Dona.': 'Other', 'Lady.': 'Other',
       'Major.': 'Military', 'Mlle.': 'Other', 'Ms.': 'Other', 'Col.': 'Military', np.nan: 'Other'})

#age_fare interaction
merged_data['age_fare'] = 1/merged_data.Age * merged_data.Fare

#Husbands name
merged_data['maiden_name'] = (merged_data.Name.str.contains("\\(")) * 1

#No fare
merged_data['no_fare'] = (merged_data.Fare == 0) * 1

#Impute the age variable based on the title
merged_data['adjusted_age'] = merged_data.Age
age_na = merged_data.Age.isna()
for title in merged_data.title:
    title_avg_age = merged_data.Age[(~age_na) & (merged_data.title == title)].median()
    merged_data.loc[(merged_data.title == title) & (age_na), 'adjusted_age'] = title_avg_age

#Make age group and fare groups
merged_data['age_bin'] = pd.qcut(merged_data['adjusted_age'], 5)
merged_data['fare_bin'] = pd.qcut(merged_data['Fare'], 5)

age_bin_sorted = list(merged_data['age_bin'].value_counts().index.sort_values().astype(str))
fare_bin_sorted = list(merged_data['fare_bin'].value_counts().index.sort_values().astype(str))

merged_data['age_bin'] = merged_data['age_bin'].astype(str).replace(age_bin_sorted, [0, 1, 2, 3, 4])
merged_data['fare_bin'] = merged_data['fare_bin'].astype(str).replace(fare_bin_sorted, [0, 1, 2, 3, 4])

#Resplit data
train_data = merged_data[~merged_data.PassengerId.isin(test_ids)]
test_data = merged_data[merged_data.PassengerId.isin(test_ids)]

test_data = test_data.drop(['Survived'], axis = 1)

#subclass of class
train_data['fare_class_diff'] = train_data['Fare'] - train_data.groupby('Pclass')['Fare'].transform('mean')
test_data['fare_class_diff'] = test_data['Fare'] - test_data.groupby('Pclass')['Fare'].transform('mean')

#Free up memory
del(merged_data)
