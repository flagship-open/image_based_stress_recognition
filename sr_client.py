# -*- coding: utf-8 -*-
"""This module is used by the client calling the API."""
import requests

VIDEO_PATH = 'APITestData/AT-001-01_01.mp4'
IP_ADDRESS = 'http://localhost:5000/predict'

sess = requests.Session()
r = sess.post(IP_ADDRESS, data={'video_path': VIDEO_PATH})

if r.status_code == 200:
    print(r, 'success')
    print(r.content)
else:
    print('fail :(')
