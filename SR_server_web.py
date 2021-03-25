from flask import Flask, json, request
from SR_server_code import StressRecognition
from collections import OrderedDict
import os
import glob
import numpy as np
import sys
from flask import Flask, json, Response, request, jsonify
from collections import OrderedDict
# from voice_speaker_recognition import voice_registration, voice_identification, voice_gallery
import operator
from flask_restplus import Resource, Api, fields

app = Flask(__name__)
api = Api(app, version='1.0', title='Image Based Stress Recognition', description='영상기반 스트레스 인식')
ns = api.namespace('Stress_Recognition_Request', description='영상기반 스트레스 인식 결과')
app.config.SWAGGER_UI_DOC_EXPANSION = 'full'

model_input = api.model('input',{'video_path':fields.String(required=True, description='비디오 저장 경로')})

@ns.route('/')
class SR(Resource):
    @ns.expect(model_input)
    def post(self):
        path = request.json['video_path']
        result = OrderedDict()

        sr_result = StressRecognition(path).data.tolist()

        # stress recognition result 를 출력하는 ID는 100004
        # 결과값이 0이 나오면 no stress, 1이 나오면 weak stress, 2가 나오면 strong stress
        result["100004"] = sr_result
        print(result)
        print('Done!!')

        return result


if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=True) # -> 동일 컴퓨터에서 요청을 보낼 때 사용
    # app.run(host='165.132.121.196', debug=True) # -> 다른 컴퓨터에 요청을 보낼 때 사용, 중간에 ip 주소 입력
