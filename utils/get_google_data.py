import json
import os
from io import BytesIO
from urllib import request

from PIL import Image

# Parameters:
ZOOM = 18
WIDTH = 400
HEIGHT = 425
LOGO_HEIGHT = 25
SAT = "satellite"
ROAD = "roadmap"
assert os.path.exists('api_key.json'), "You need to provide your own Google API and Map key in api_key.json."
with open('api_key.json') as f:
    keys = json.load(f) # read api and map key from json file
API_KEY = keys['api_key']
MAP_ID = keys['map_id']


def download_google_data(cities, expansion, stepsize):
    """
    Downloads satellite images and segmentation maps from Google Maps API into google_data folder.
    
    Args:
        cities (list): List of tuples of (latitude, longitude) coordinates.
        expansion (int): Number of steps to expand around each city.
        stepsize (float): Distance between two image centers in degrees.
    """
    # Create data folder
    if not os.path.exists("google_data"):
        os.makedirs("google_data")
        os.makedirs("google_data/images")
        os.makedirs("google_data/groundtruth")

    # expand centers
    centers = []
    for city in cities:
        for i in range(-expansion,expansion+1):
            for j in range(-expansion,expansion+1):
                centers.append((city[0] + i * stepsize, city[1] + j * stepsize))

    index = 0 #starting image index
    for center in centers:
        # construct urls
        url_sat = "https://maps.googleapis.com/maps/api/staticmap" \
                    + f"?zoom={ZOOM}" \
                    + f"&size={WIDTH}x{HEIGHT}" \
                    + f"&maptype={SAT}" \
                    + f"&center={center[0]},{center[1]}" \
                    + f"&key={API_KEY}"
        url_seg = "https://maps.googleapis.com/maps/api/staticmap" \
                    + f"?zoom={ZOOM}" \
                    + f"&size={WIDTH}x{HEIGHT}" \
                    + f"&maptype={ROAD}" \
                    + f"&center={center[0]},{center[1]}" \
                    + f"&key={API_KEY}" \
                    + f"&map_id={MAP_ID}"

        filename = f"satimage_{index}.png"
        filename_sat = "google_data/images/" + filename
        filename_seg = f"google_data/groundtruth/" + filename

        # download sattelite images if they don't exist
        if(not os.path.exists(filename_sat)):
            response_sat = request.urlopen(url_sat)
            image_sat = Image.open(BytesIO(response_sat.read()))
            image_sat = image_sat.crop((0, 0, WIDTH, HEIGHT - LOGO_HEIGHT))
            image_sat.save(filename_sat)
        
        # download segmentation maps if they don't exist
        if(not os.path.exists(filename_seg)):
            response_seg = request.urlopen(url_seg)
            image_seg = Image.open(BytesIO(response_seg.read()))
            image_seg = image_seg.crop((0, 0, WIDTH, HEIGHT - LOGO_HEIGHT))
            image_seg.save(filename_seg)
        
        index += 1


if __name__ == "__main__":
    """
    Download satellite images and segmentation maps for LA and New Jersey.

    A stepsize of 0.002 corresponds to a distance of 222m between two image centers.
    An expansion of 19 corresponds to a total of 1521 images per city.
    """

    cities = [
        (33.98774563224179, -118.24251823835559), # LA
        (40.89672, -74.1857) # New Jersey
    ]

    download_google_data(cities=cities, expansion=19, stepsize=0.002)