from bs4 import BeautifulSoup
import requests 
import pandas as pd
import time

data = pd.read_excel("Data\\joined_housing_data.xlsx")

subset_data = data[['address', 'MLS_num']]

domain = 'https://torontocentral.evrealestate.com/ListingDetails/'

length = len(data)
new_data = []

for i, row in subset_data.iterrows():

    if (i+1) % 100 == 0:
        print("Scraping " + str(i+1) + "th URL of " + str(length) + " URLs")

    locator = row['address'].replace(' ', '-') + '/' + row['MLS_num']
    url = domain + locator

    #Get url
    try:
        response = requests.get(url)
        html = response.content
        soup = BeautifulSoup(html, 'html.parser')
    except:
        continue
    
    #Get sold date
    try:
        sold_date = soup.select_one('div.rng-listing-details-main-information-market-estimate > span').text
    except:
        continue

    #Get other details. Only 'try' first one because all other selectors will appear if first one does.
    try:
        median_age = soup.select_one('i.rni-median-age + div').contents[0].replace('\r', '').strip()
    except:
        continue

    median_income = soup.select_one('i.rni-estimated-median-income + div').contents[0].replace('\r', '').strip()
    owner_divide = soup.select_one('i.rni-estimated-owner-renter-divide + div').contents[0].replace('\r', '').strip()
    pop_density = soup.select_one('i.rni-estimated-population-density + div').contents[0].replace('\r', '').strip()
    total_owners = soup.select_one('i.rni-estimated-total-home-owners + div').contents[0].replace('\r', '').strip()
    hh_size = soup.select_one('i.rni-estimated-average-household-size + div').contents[0].replace('\r', '').strip()
    college_educated = soup.select_one('i.rni-educational-attainment + div').contents[0].replace('\r', '').strip()
    total_renters = soup.select_one('i.rni-estimated-total-renters + div').contents[0].replace('\r', '').strip()
    total_pop = soup.select_one('i.rni-total-number-of-people + div').contents[0].replace('\r', '').strip()

    #Add to list and sleep
    new_data.append([row['MLS_num'], sold_date, median_age,  median_income, owner_divide, pop_density, 
                    total_owners, hh_size, college_educated, total_renters, total_pop])
    
    time.sleep(1.5)

output = pd.DataFrame(new_data, columns = ['MLS_num', 'sold_date', 'median_age',  'median_income', 'owner_renter_divide', 'pop_density', 
                                           'total_owners', 'hh_size', 'college_educated', 'total_renters', 'total_pop'])

output.to_excel('engalvolk_output1.xlsx', index = False)

output.drop_duplicates()
excel_data = data.merge(output.drop_duplicates(), how = 'left', left_on = 'MLS_num', right_on = 'MLS_num')

excel_data.to_excel('new_joined_housing_data.xlsx', index = False)
