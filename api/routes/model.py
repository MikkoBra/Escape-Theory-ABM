from flask import Blueprint, jsonify, request
from network_abm.models.EscapeModel import EscapeModel
from Constants import MINUTE_LENGTH

model_bp = Blueprint("model", __name__)

_model_instance = None

@model_bp.route("/initialize", methods=["POST"])
def initialize():
    global _model_instance

    data = request.get_json() or {}
    N  = data.get("n_agents", 15)
    dt = data.get("dt", MINUTE_LENGTH)
    end_time = data.get("end_time", 10)

    _model_instance = EscapeModel(dt=dt, end_time=end_time, n_agents=N,)

    return jsonify({
        "status": "ok",
    }), 201

@model_bp.route("/step", methods=["POST"])
def step():
    
    global _model_instance
    if _model_instance is None:
        return jsonify({"error": "model not initialized"}), 400

    _model_instance.debug = True
    _model_instance.step()
    _model_instance.debug = False

    return jsonify({
        "status": "ok",
        "t": _model_instance.time,
    }), 200

@model_bp.route("/run", methods=["POST"])
def run_model():
    if _model_instance is None:
        return jsonify({"error": "model not initialized"}), 400
    
    while _model_instance.steps < _model_instance.end_steps:
        _model_instance.step()
    
    stress = _model_instance.datacollector["Stress"].tolist()
    return jsonify({
        "stress": stress,
    }), 200