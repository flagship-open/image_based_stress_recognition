from flask import Flask, json, request
from SR_server_code import StressRecognition
from collections import OrderedDict


app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":

        path = request.form['video_path']
        result = OrderedDict()

        SR_output, FR_feature = StressRecognition(path).data.tolist()

        # stress recognition result ID: 100004
        # 0 -> no stress, 1 -> weak stress, 2 -> strong stress
        result["100004"] = SR_output
        # face recognition feature ID: 100005
        # 128 dimension feature, float32
        result["100005"] = FR_feature
        print(result)
        print('Done!!')

        return json.dumps(result)


if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=True) # -> using when requesting same computer
    # app.run(host='165.132.120.198', debug=True) # -> using when requesting different computer, input IP address
