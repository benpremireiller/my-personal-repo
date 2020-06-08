import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from scipy.stats import ttest_ind

def split_na_data(x, y):
    condition = x.isna()
    without_na = y[~condition]
    with_na = y[condition]
    return without_na, with_na

train_data = pd.read_csv('.\\Data\\Titanic\\train.csv')
test_data = pd.read_csv('.\\Data\\Titanic\\test.csv')

train_copy = train_data
merged_data = pd.concat([train_data, test_data])

#Missing data for all variables
train_data.isnull().sum()

#Missing age variables
with_age, without_age =  split_na_data(train_data.Age, train_data.Survived)
ttest_ind(with_age, without_age)

#Corr matrix
train_data.corr()
#Is there a pattern in the ticket number?

#Investigate people who didn't pay fare
train_data[train_data.Fare == 0] #14/15 dead
not_free, free =  split_na_data(train_data.Fare.replace(0, np.nan), train_data.Survived)
ttest_ind(free, not_free)

#How many names have brackets (looks like maiden name) in them
train_copy['maiden_name'] = np.where(train_data.Name.str.contains("\\("), 'Yes', 'No')
train_copy.groupby('maiden_name')['Survived'].mean()

#Survival by cabin_deck and cabin raw number
train_copy['cabin_deck'] = train_data.Cabin.str.extract('(^\w)')[0]
train_copy['cabin_raw_number'] = train_data.Cabin.str.extract('(^\w)(\d+$)')[1]
train_copy.groupby('cabin_deck').apply(lambda g: g.Survived.sum()/g.Survived.count())

#Plot deck by survive
plot = sns.catplot(x="cabin_deck", y="Survived", hue="Sex", kind="bar", data=train_data,
                    order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']).set_ylabels("survival probability")
plot = sns.catplot(x="cabin_deck", y="Survived", hue="Pclass", kind="bar", data=train_data).set_ylabels("survival probability")

#Find cabin counts by pclass
train_copy['has_cabin'] = (train_copy.Cabin.notna()) * 1
plot = sns.catplot(x="has_cabin", hue="Pclass", kind="count", data=train_data)

#Plot has cabin by sex
plot = sns.catplot(x="has_cabin", y="Survived", hue="Sex", kind="bar", data=train_copy).set_ylabels("survival probability")

#Decompose ticket
train_copy['ticket_prefix'] = train_copy.Ticket.str.extract('(.+ |[a-zA-Z]+)?(\d*)')[0].str.strip()
train_copy['ticket_number'] = train_copy.Ticket.str.extract('(.+ |[a-zA-Z]+)?(\d*)')[1]
train_copy['ticket_len'] = train_copy.ticket_number.apply(len)

#Ticket length by survived
train_copy.groupby('ticket_len')['Survived', 'Fare'].agg([('Ratio', 'mean'), ('Count of passengers', 'count')])

#Distribution of fare by ticket len
ticket_lengths = [4,5,6,7]
for length in ticket_lengths:
    len_data = train_copy[train_copy.ticket_len == length].Fare
    np.log(len_data[len_data != 0]).plot(kind = 'density', label = length)
plt.title('Fare dist by ticket length')
plt.legend()

pd.crosstab(train_copy.ticket_len, train_data.Embarked)
pd.crosstab(train_copy.ticket_len, train_data.Pclass)

plot = sns.catplot(x="ticket_len", y="Survived", hue="Pclass", kind="bar", data=train_copy[train_copy.ticket_len.isin([4,5,6,7])]).set_ylabels("survival probability")

#Look at fare***************************************

#Inspect fare rate by survival
fig, ax = plt.subplots(1,1)
for t in [0, 1]:
    survive_data = train_copy[train_copy.Survived == t].Fare
    np.log(survive_data[survive_data != 0]).plot(kind = 'density', label = t)
plt.title('Fare dist by survive')
plt.legend()

#Inspect fare rate by survival and maybe cabin type
fig, ax = plt.subplots(1,1)
for t in train_copy.Pclass.unique():
    class_data = train_copy[train_copy.Pclass == t].Fare
    np.log(class_data[class_data != 0]).plot(kind = 'density', label = t)
plt.title('Fare dist by class')
plt.legend()


#Inspect prefixes
train_copy.ticket_prefix.value_counts()
train_copy.groupby('ticket_prefix')['Survived'].agg(['mean', 'count']).sort_values('mean', ascending = False)

#First digit of ticket number
train_copy['ticket_first'] = train_copy['ticket_number'].str.extract('(^\d{1})')[0]
train_copy.groupby(['ticket_first'])['Survived', 'Fare'].agg(['mean', 'count'])

train_copy.groupby(['ticket_len', 'Pclass'])['Survived', 'Fare'].agg(['mean', 'count'])
train_copy.groupby(['ticket_len', 'ticket_first'])['Survived', 'Fare'].agg(['mean', 'count'])
train_copy.groupby(['Pclass', 'ticket_first'])['Survived', 'Fare'].agg(['mean', 'count']) #Are they just subclasses?
pd.crosstab(train_copy.ticket_first, train_data.Pclass)

#Check survival rate of people with an without cabins
with_cabin, without_cabin = split_na_data(train_copy.cabin_deck ,train_copy.Survived)
with_cabin.mean()
without_cabin.mean()

#Gender survival rate
train_copy.groupby('Sex').apply(lambda g: g.Survived.sum()/g.Survived.count())

#Plot of survival by class and gender
plot = sns.catplot(x="Sex", y="Survived", hue="Pclass", kind="bar", data=train_data).set_ylabels("survival probability")

#Dummie for class
train_class = pd.get_dummies(train_copy, columns = ['Pclass','Embarked'])
train_copy.corr()

#Inspect age survival
bins = [*range(0, int(max(train_copy.Age)+1), 20)]
train_copy['Age_groups'] = pd.cut(train_copy.Age, bins)
plot = sns.catplot(x="Age_groups", y="Survived", hue="Pclass", kind="bar", data=train_data).set_ylabels("survival probability")
plot = sns.catplot(x="Age_groups", y="Survived", hue="Sex", kind="bar", data=train_data).set_ylabels("survival probability")

#Inspect embarkment survival
plot = sns.catplot(x="Embarked", hue="Pclass", kind="count", data=train_data).set_ylabels("survival probability")
plot = sns.catplot(x="Embarked", y="Survived", hue="Pclass", kind="bar", data=train_data).set_ylabels("survival probability")



