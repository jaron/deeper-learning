import math
import requests
import csv
import os

# configure these parameters
base_url = 'http://api.zoopla.co.uk/api/v1/property_listings.js'
data_limit = 200
page_size = 10

# geographical parameters

# the postcode of interest
area = "IP5"
# the location of this postcode's nearest mainline train station, (latitude, longitude)
nearest_station = (52.051, 1.144)

params = {
    "api_key" : "jj7k6du6z3bamad4jket3ny3",
    "area" : area,
    "include_sold" : "1",
    "listing_status" : "sale",
    "page_size" : page_size
}


def calculate_distance(loc1, loc2):
    R = 6371.0 # approximate radius of earth in km

    lat1 = math.radians(loc1[0])
    lon1 = math.radians(loc1[1])
    lat2 = math.radians(loc2[0])
    lon2 = math.radians(loc2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return round(distance, 3)


def number_of_garages(description):
    num_garages = 0
    if ("garage" in description):
        num_garages = 1
    if ("double garage" in description or "double detached garage" in description or "two garages" in description):
        num_garages = 2
    if ("treble garage" in description or "triple garage" in description):
        num_garages = 3
    return num_garages


def number_of_bathrooms(description, listing):
    num_bathrooms = max(1, int(listing.get("num_bathrooms")))
    if ("two bathroom" in description):
        num_bathrooms = 2
    if ("three bathroom" in description):
        num_bathrooms = 3
    return num_bathrooms


def get_type(type, description):
    if "bungalow" in type.lower() or "bungalow" in description: return "Bungalow"
    if "semi-detached" in type.lower(): return "Semi"
    if "detached" in type.lower(): return "Detached"
    if "town house" in type.lower(): return "Detached"
    if "terrace" in type.lower(): return "Terraced"
    if "mobile" in type.lower(): return "Mobile"
    if "apartment" in description: return "Flat"

    print "Unable to determine type: ", type, " - ", description
    return "Unknown"


def extract_details(listing):
    # determine the features we're interested in keeping
    # remember, we can only use continuous values, so categorical features like property_type will need to be converted
    # as the estimator assumes a natural ordering between values

    description = listing.get("short_description").lower() + " " + listing.get("description").lower()
    price = int(listing.get("price"))

    # remove properties without a price, or miscategorised rental properties
    if price < 10000:
        return False

    results = []
    results.append(price)
    results.append(int(listing.get("num_bedrooms")))
    results.append(number_of_bathrooms(description, listing))
    results.append(number_of_garages(description))
    results.append(get_type(listing.get("property_type"), description))

    # add distance to transport hub, and locations
    location = (float(listing.get("latitude")), float(listing.get("longitude")))
    distance = calculate_distance(nearest_station, location)
    results.append(listing.get("latitude"))
    results.append(listing.get("longitude"))
    results.append(distance)

    print (results)
    return results


def get_filename():
    filename = "../data/{0}-properties.csv".format(area)
    return filename


def write_to_csv(buffer):
    filename = get_filename()
    print "Writing to:", filename

    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"')
        csvwriter.writerows(buffer)
        csvfile.flush()


def get_details(page_number, counter=0):
    params["page_number"] = page_number
    response = requests.get(base_url, params)
    print (response.url)
    data = response.json()
    result_count = data.get("result_count")

    print (result_count)

    buffer = []
    listings = data.get("listing")
    for i in range(len(listings)):
        row = extract_details(listings[i])
        if not row:
            print "Skipped entry ", i
        else:
            buffer.append(row)

    write_to_csv(buffer)

    counter += len(listings)
    print ("After: Counter = {0}".format(counter))
    return result_count, counter


def main():
    counter = 0
    need_more_data = True
    page_number = 0

    filename = get_filename()
    print "Deleting file:", filename
    os.remove(filename)

    header = ["Price","Bedrooms","Bathrooms","Garages","Type","Latitude","Longitude","Remoteness"]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"')
        csvwriter.writerow(header)
        csvfile.flush()

    while need_more_data:
        page_number += 1
        num_available, counter = get_details(page_number, counter)
        need_more_data = num_available > 0 and counter < num_available and counter < data_limit
        print ("Fetched {0} records".format(counter))


main()