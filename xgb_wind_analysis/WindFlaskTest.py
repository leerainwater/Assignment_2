#
# Author: Lee Rainwater
# Borrowed Heavily From: Jamey Johnston
# Title: Test Flask App to Docker
# Date: 2021/02/21
# Email: lee.rainwater@tamu.edu
# Texas A&M University - MS in Analytics - Mays Business School
#

import requests, json

# url = 'http://127.0.0.1:5000/' # Local Flask Test
url = 'http://127.0.0.1:8080/' # Docker Test

# "alt", "lat", "mo"
# valid values:
#   50.0 <= alt <= 100.0 (altitude in km, float)
#   0.0 <= lat <= 60.0 (latitude, float)
#   1 <= mo <= 12 (month, integer)
text = json.dumps({"0":{"alt":55.0, "lat":40.0, "mo":7},
               	"1":{"alt":55.0, "lat":20.0, "mo":6},
               	"2":{"alt":65.0, "lat":30.0, "mo":12}
                })

headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

r = requests.post(url, data=text, headers=headers)

print(r,r.text)


    # Predict a altitude diversity from inputs
    # alt, lat, mo = (55, 20, 6)
    # alt_div = loaded_modelRF.predict([[alt, lat, mo]])

    # print(f'Using altitude {alt}kft, latitude {lat}, month {mo}')
    # print(f'Predicted altitude diversity: {alt_div[0]:.3f}')
