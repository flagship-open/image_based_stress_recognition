import requests
import glob

ip_address = 'http://165.132.121.196:5000/predict' # -> 다른 컴퓨터에 요청을 보낼 때 사용, 중간에 ip 주소 입력
sess = requests.Session()

no_stress_response = 0
weak_stress_response = 0
strong_stress_response = 0

for video_path in glob.glob('APITestData/*.mp4'):
    r = sess.post(ip_address, data={'video_path': video_path})
    gt = int(video_path.split('AT-001-0')[-1][0])

    if gt == 1:
        if r.status_code == 200:
            no_stress_response += 1
    elif gt == 2:
        if r.status_code == 200:
            weak_stress_response += 1
    elif gt == 3:
        if r.status_code == 200:
            strong_stress_response += 1

print("AT-001-01 -> Result: {}/25 (correct/all)".format(no_stress_response))
print("AT-001-02 -> Result: {}/25 (correct/all)".format(weak_stress_response))
print("AT-001-03 -> Result: {}/25 (correct/all)".format(strong_stress_response))
