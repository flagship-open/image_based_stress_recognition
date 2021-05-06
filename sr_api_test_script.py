# -*- coding: utf-8 -*-
"""This is a script module that tests api."""
import glob
import requests

IP_ADDRESS = 'http://165.132.121.196:5000/predict'  # 다른 컴퓨터에 요청을 보낼 때 사용, 중간에 ip 주소 입력
sess = requests.Session()

NO_STRESS_RESPONSE = 0
WEAK_STRESS_RESPONSE = 0
STRONG_STRESS_RESPONSE = 0

for video_path in glob.glob('APITestData/*.mp4'):
    r = sess.post(IP_ADDRESS, data={'video_path': video_path})
    gt = int(video_path.split('AT-001-0')[-1][0])

    if gt == 1:
        if r.status_code == 200:
            NO_STRESS_RESPONSE += 1
    elif gt == 2:
        if r.status_code == 200:
            WEAK_STRESS_RESPONSE += 1
    elif gt == 3:
        if r.status_code == 200:
            STRONG_STRESS_RESPONSE += 1

print("AT-001-01 -> Result: {}/25 (correct/all)".format(NO_STRESS_RESPONSE))
print("AT-001-02 -> Result: {}/25 (correct/all)".format(WEAK_STRESS_RESPONSE))
print("AT-001-03 -> Result: {}/25 (correct/all)".format(STRONG_STRESS_RESPONSE))
