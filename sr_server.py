# -*- coding: utf-8 -*-
"""This is the module that runs the API server."""
from collections import OrderedDict
from flask import Flask, json, request
from sr_server_code import stress_recognition


app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """This function recognizes the stress of the video input from the client."""
    if request.method == "POST":
        path = request.form['video_path']
        result = OrderedDict()

        sr_output, fr_feature = stress_recognition(path).data.tolist()

        # stress recognition result ID: 100004
        # 0 -> no stress, 1 -> weak stress, 2 -> strong stress
        result["100004"] = sr_output
        # face recognition feature ID: 100005
        # 128 dimension feature, float32
        result["100005"] = fr_feature
        print(result)
        print('Done!!')

        return json.dumps(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True) # using when requesting same computer

    # using when requesting different computer, input IP address
    # app.run(host='165.132.121.196', debug=True)
