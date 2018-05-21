from flask import jsonify, request
from app.api import bp
from app.logistic_regression_trainer import train_model
import json


@bp.route('/update_model', methods=['POST'])
def update_model():
    data = request.get_json() or {}
    success = train_model(json.dumps(data))
    respdata = {'success': success}
    response = jsonify(respdata)
    response.status_code = 200

    filename = 'model/model_graph.pb'

    return get_bytes_from_file(filename)


def get_bytes_from_file(filename):
    return open(filename, "rb").read()