from geopy.extra.rate_limiter import RateLimiter
import geopy
from geopy import Nominatim
from geopy import Photon
from geopy import distance
import psycopg2
from psycopg2 import Error
import pandas as pd
import os
import numpy as np

def load_to_db(data):
    try:
        connection = psycopg2.connect(user = "postgres",
                                    password = "password",
                                    host = "127.0.0.1",
                                    port = "5432",
                                    database = "postgres")

        cursor = connection.cursor()

        #Insert new values into table
        insert_query = """INSERT INTO housing.locations VALUES
         (%s, %s, %s, %s, %s, %s)
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


data = pd.read_excel("Data\\new_housing_data.xlsx")

#Pull subway names
subway_stations = pd.read_excel('Data\\subway_stations.xlsx', header = None)
city = ', Toronto, Ontario, Canada'

#Set up geocoder
geopy.geocoders.options.default_timeout = None
locator = Nominatim(user_agent='housingProject-toronto')
geocode = RateLimiter(locator.geocode, min_delay_seconds=3)

#Pull addresses
full_addresses = data.address + city
house_locations = []
for i, address in enumerate(full_addresses):
    if (i + 1) % 50 == 0:
        print(i)
    house_locations.append(geocode(address))

#Pull subway stations
station_names = subway_stations[0]
full_station_names = station_names + city
subway_locations = [geocode(subway) for subway in full_station_names]

subway_locations = [sub for sub in subway_locations if sub is not None]

#Export subway data
pd.DataFrame(subways, columns = ['name', 'coordinates']).to_excel('Data\\subway_coordinate_data.xlsx')

#Find smallest distance between addresses and subway stations
merged_location_data = []

for house_location in house_locations:
    if not house_location:
        merged_location_data.append([None for i in range(5)]) #No data found
        continue
    
    #Get region info and latitude longitude coordinates
    house_region = [*map(str.strip, location.address.split(",")[2:5])]
    house_coordinates = location.point[:2]
    station_house_distances = []

    for subway_location in subway_locations:
        if not subway:
            continue
        name = subway_location.address.split(",")[0].replace(")",'').replace("(",'')
        station_coordinates = subway_location.point[:2]
        dist = distance.distance(house_coordinates, station_coordinates).km
        station_house_distances.append([dist, name])

    min_distance = min(station_house_distances, key = lambda x: x[0])
    merged_location_data.append(min_distance + house_region  + [house_coordinates[0]] + [house_coordinates[1]])

merged_location_data = pd.DataFrame(merged_location_data, columns = ['subway_dist', 'subway_name',
                                                            'neighborhood', 'district', 'area',
                                                             'latitude', 'longitude'])
merged_location_data.insert(loc=0, column='MLS_num', value=data.MLS_num)

merged_location_data.subway_dist = merged_location_data.subway_dist.astype('str')

merged_location_data.to_excel("Data\\second_batch_location_data.xlsx")

#load_data = [tuple(x) for x in merged_location_data.values]
#load_to_db(load_data)

#Merge with existing dataframe
#TODO


