# -*- coding: utf-8 -*-
"""This is a module with functions used in the server."""

from collections import OrderedDict
from flask import Flask, request
from flask_restplus import Resource, Api, fields
from sr_server_code import stress_recognition

app = Flask(__name__)
api = Api(app, version='1.0', title='Image Based Stress Recognition', description='영상기반 스트레스 인식')
ns = api.namespace('Stress_Recognition_Request', description='영상기반 스트레스 인식 결과')
app.config.SWAGGER_UI_DOC_EXPANSION = 'full'

model_input = api.model('input', {'video_path': fields.String(required=True,
                                                              description='비디오 저장 경로')})

@ns.route('/')
class SR(Resource):
    """This is a class that recognizes stress."""
    @ns.expect(model_input)
    def post(self):
        """This is a function that posts a video."""
        path = request.json['video_path']
        result = OrderedDict()

        sr_result = stress_recognition(path).data.tolist()

        # stress recognition result 를 출력하는 ID는 100004
        # 결과값이 0이 나오면 no stress, 1이 나오면 weak stress, 2가 나오면 strong stress
        result["100004"] = sr_result
        print(result)
        print('Done!!')

        return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)  # using when requesting same computer

    # using when requesting different computer, input IP address
    # app.run(host='165.132.121.196', debug=True)
