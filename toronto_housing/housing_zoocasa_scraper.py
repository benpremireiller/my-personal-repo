from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import requests 
import psycopg2
from psycopg2 import Error
import pandas as pd
import datetime as dt
import time
import re

""" ****NOTE****
This script currently only works to scrape SOLD HOUSES from zoocasa. It may
be adapted in the future to include townhouses and condos/apartments as well 
as active listings.
Note: You need an account for this scraper to work
Note: All data collected will be have been posted within 6 months of scrape date (Zoocasa policy)
"""

def log_in():
    
    email = "email"
    password = "password"

    driver = webdriver.Firefox()

    #Log in
    driver.get("https://www.zoocasa.com/")

    try:
        sign_in_button = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, "//a[@role='button' and text()='Sign in']")))
        is_signed_in = False
    except NoSuchElementException:
        is_signed_in = True    

    #Sign in because cannot view houses without sign in
    if not is_signed_in:
        sign_in_button.click()
        driver.find_element_by_xpath("//input[@type='email']").send_keys(email)
        driver.find_element_by_xpath("//input[@type='password']").send_keys(password)
        driver.find_element_by_xpath("//button[text()='Sign in']").click()
    
    return driver


###Get all listings from 'send_key' city
def get_urls(driver, send_key, short_wait = 1.5):

    #Pull first page of houses from 'send_key' city
    driver.get("https://www.zoocasa.com/{}-sold-listings".format(send_key))

    home_type_list = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH,"//drop-down[contains(@class, 'home-type-drop-down')]")))
    home_type_list.click()

    #Ensure houses are selected (this may not work because one button always has to be selected)
    house_button = home_type_list.find_element_by_xpath(".//span[@role='button' and text()='House']")
    townhouse_button = home_type_list.find_element_by_xpath(".//span[@role='button' and text()='Townhouse']")
    condo_apartment_button = home_type_list.find_element_by_xpath(".//span[@role='button' and text()='Condo/Apartment']")

    #Check if buttons are selected and adjust
    if not house_button.get_attribute('class') == 'active':
        house_button.click()
    if townhouse_button.get_attribute('class') == 'active':
        townhouse_button.click()
    if condo_apartment_button.get_attribute('class') == 'active':
        condo_apartment_button.click()

    #Apply changes
    home_type_list.find_element_by_xpath(".//button[@class='apply']").click()

    #Loop through houses on page
    print('Collecting URLs for: '+ send_key)
    
    href_list = []
    page_no = 1

    while True:

        #If there is a right arrow, there are more pages to search
        right_arrow = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.XPATH, "//a[contains(@class, 'icon-arrow-right-open')]")))

        if page_no % 5 == 0:
            print("Currently scraping page: " + str(page_no))

        #Find url for all listings on current page
        listings = driver.find_elements_by_xpath("//loading-data/*")

        for i in range(len(listings)-1):
            current_listing = listings[i].find_element_by_xpath('.//a').get_attribute('href')
            href_list.append(current_listing)
        
        #End loop if at the end of the listings for the current 'send_key' city
        right_arrow_exists = right_arrow.get_attribute('href')
        if not right_arrow_exists:
            break

        page_no += 1
        time.sleep(short_wait)
        right_arrow.click()

    href_list = list(set(href_list)) #Remove dups: Zoocasa lists houses more than once

    print('Total counts of houses: {}'.format(len(href_list)))

    return href_list

###Fetch housing data from each listing

def get_data(href_list, key, wait_interval = 1.5, message_freq = 25):

    if type(href_list) != list:
        raise('href_list must be of type list')

    housing_data = []
    length = len(href_list)

    for i, href in enumerate(href_list): 

        house_data = []

        if (i+1) % message_freq == 0:
            print("Scraping " + str(i+1) + "th URL of " + str(length) +" URLs")
            print("Time remaining: at least "+ str(round((wait_interval*(length-(i+1)))/60,1)) + " minutes")

        #Load each URL
        try:
            response = requests.get(href)
            html = response.content
            soup = BeautifulSoup(html, 'html.parser')
        except:
            continue

        #!First portion of data is seperated and must be extracted manually!#
        #Street address (try first element incase page is not valid or 404)
        try:
            house_data.append(soup.select_one("span[itemProp=streetAddress]").text)
        except:
            continue
        
        #List price
        house_data.append(soup.select_one("div.list-price > span:nth-of-type(1)").text.replace("$","").replace(",","")) 
        #Sold Price
        house_data.append(soup.select_one("div.sold-price > span:nth-of-type(1)").text.replace("$","").replace(",",""))
        #Number of beds
        house_data.append(soup.select_one("div.beds-baths > :nth-of-type(1)").text.replace(" beds",""))
        #Number of baths
        house_data.append(soup.select_one("div.beds-baths > :nth-of-type(2)").text.replace(" baths",""))
        #Number of parking spaces
        house_data.append(soup.select_one("div.beds-baths > :nth-of-type(4)").text.replace(" parking",""))
        #Description
        house_data.append(soup.select_one("p.description").text)

        #!At this point the remainder of the data is in tables!#
        #Loop through detail tables to extract info
        tables = soup.select("details-table")

        #Declare rooms sqft, name and floor lists to append to when we reach the 4th table of loop below
        rooms_sqft = []
        room_names = []
        room_floors = []

        for i, table in enumerate(tables):
            #Get row elements
            table_rows = table.select("section > div")

            #Append the table values to the 'housing_data' list   
            for table_row in table_rows:

                #Add all the values from first 3 tables as is.
                if i < 3:
                    try:
                        house_data.append(table_row.select_one("span:nth-of-type(2)").text)
                    except:
                        house_data.append("Error")

                #Stop at the room sqft table because it is the last (4th) table to be scraped.
                elif i == 3:

                    #Calculate total square footage by summing all sqft of each room and find room names/room floor
                    try:
                        room_sqft = table_row.select_one("span:nth-of-type(3)").text.replace("ft","").replace("×", "*").replace("\n","").strip()
                        room_floor = table_row.select_one("span:nth-of-type(2)").text
                        room_name =  table_row.select_one("span:nth-of-type(1)").text
                    except:
                        room_sqft = 'Error'
                        room_floor = 'Error'
                        room_name = 'Error'
                        break

                    #Some rooms don't have sq ft values or have errors. Only append those correct values.
                    is_proper_format = re.findall('\d+\.*\d* *\* *\d+\.*\d*', room_sqft)
                    if is_proper_format:
                            rooms_sqft.append(eval(room_sqft))
                    else:
                        rooms_sqft.append(0)
                    
                    #Append name and floor
                    room_names.append(room_name)
                    room_floors.append(room_floor)

                #The last table (5th) is the extras
                else:
                    house_data.append(table_row.select_one("div > p").text) #TODO: add new column in sql db for this


        #Add total sqft, number of rooms, room floors and room names respectively
        house_data.append(round(sum(rooms_sqft)))
        house_data.append(len(rooms_sqft)) #TODO: add new column in sql db for this
        house_data.append(', '.join(room_floors)) #TODO: add new column in sql db for this
        house_data.append(', '.join(room_names)) #TODO: add new column in sql db for this

        #Add the city, current date and URL
        house_data.append(key) 
        house_data.append(dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%d"))
        house_data.append(href) #TODO: add new column in sql db for this

        #Append and wait shot_delay seconds to not overload server
        housing_data.append(tuple(house_data))
        time.sleep(wait_interval)
    
    return housing_data

def get_housing_data(keys):

    if type(keys) != list:
        raise ValueError("Keys must be of type list")

    data = []

    for key in keys:
        driver = log_in()
        hrefs = get_urls(driver, key)
        driver.close()
        data += get_data(hrefs, key)

    return data

def load_to_db(data):
    try:
        connection = psycopg2.connect(user = "postgres",
                                    password = "password",
                                    host = "127.0.0.1",
                                    port = "5432",
                                    database = "postgres")

        cursor = connection.cursor()

        #Insert new values into table
        insert_query = """INSERT INTO housing.housing VALUES
         (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
         %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
         %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
         %s, %s, %s, %s)
        """
        cursor.executemany(insert_query, data)

        #Seal the deal
        connection.commit()

    except (Exception, psycopg2.Error) as error :
        print ("Error while connecting to PostgreSQL", error)

    finally:
        if(connection):
            cursor.close()
            connection.close()


#Run ETL
data = get_housing_data(['toronto-on'])

#Variable names of scraped data:
names = ['address', 'list_price', 'sold_price', 'beds', 'baths', 'parking', 'description',
       'MLS_num', 'type', 'levels', 'sqft_range', 'taxes', 'laundry_level',
       'central_vac', 'fireplace', 'acreage', 'lot_size', 'exterior_type',
       'garage', 'age', 'basement_details', 'driveway', 'garage_spaces',
       'heat_type', 'AC_type', 'heating_fuel', 'extras', 'sqft_calc', 'n_rooms', 
       'room_floors', 'room_names', 'city','date_scraped', 'url']

pd.DataFrame(data, columns = names).to_excel('batch2_housing_data.xlsx')

#load_to_db(data)

