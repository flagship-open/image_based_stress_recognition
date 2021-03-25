import requests

video_path = 'APITestData/AT-001-01_01.mp4' # -> 비디오 경로
ip_address = 'http://localhost:5000/predict' # -> 동일 컴퓨터에서 요청을 보낼 때 사용
# ip_address = 'http://165.132.120.198:5000/predict' # -> 다른 컴퓨터에 요청을 보낼 때 사용, 중간에 ip 주소 입력


sess = requests.Session()
r = sess.post(ip_address, data={'video_path': video_path})

if r.status_code == 200:
    print(r,'success')
    print(r.content)
else:
    print('fail :(')
