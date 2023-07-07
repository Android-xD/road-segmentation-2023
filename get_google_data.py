import os
from io import BytesIO
from urllib import request
import json

from PIL import Image

# Parameters:
ZOOM = 18
WIDTH = 400
HEIGHT = 425
LOGO_HEIGHT = 25
SAT = "satellite"
ROAD = "roadmap"
#read api and map key from json file
# Read the contents of the JSON file
with open('api_key.json') as f:
    keys = json.load(f)
API_KEY = keys['api_key']
MAP_ID = keys['map_id']
CITIES = [
    # (34.04240, -118.45782), # LA 
    (33.98774563224179, -118.24251823835559), # LA
    (40.89672, -74.1857) # New Jersey
]
EXPANSION = 19
STEPSIZE = 0.002

# expand centers
centers = []
for city in CITIES:
    for i in range(-EXPANSION,EXPANSION+1):
        for j in range(-EXPANSION,EXPANSION+1):
            centers.append((city[0] + i * STEPSIZE, city[1] + j * STEPSIZE))

# Create data folder
if not os.path.exists("google_data"):
    os.makedirs("google_data")
    os.makedirs("google_data/images")
    os.makedirs("google_data/groundtruth")

index = 0 #starting image index
for center in centers:
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

    if(not os.path.exists(filename_sat)):
        response_sat = request.urlopen(url_sat)
        image_sat = Image.open(BytesIO(response_sat.read()))
        image_sat = image_sat.crop((0, 0, WIDTH, HEIGHT - LOGO_HEIGHT))
        image_sat.save(filename_sat)
        
    if(not os.path.exists(filename_seg)):
        response_seg = request.urlopen(url_seg)
        image_seg = Image.open(BytesIO(response_seg.read()))
        image_seg = image_seg.crop((0, 0, WIDTH, HEIGHT - LOGO_HEIGHT))
        image_seg.save(filename_seg)
    
    index += 1


