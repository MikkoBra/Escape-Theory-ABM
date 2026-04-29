from flask import Blueprint, jsonify, request
from iccs_model.networked_agents.models.NetworkedModel import NetworkedModelNumba
from Constants import MINUTE_LENGTH

model_bp = Blueprint("model", __name__)

_model_instance = None

@model_bp.route("/initialize", methods=["POST"])
def initialize():
    """Run a basic simulation with the optimized model."""
    global _model_instance

    data = request.get_json() or {}
    N  = data.get("n_agents", 15)
    dt = data.get("dt", MINUTE_LENGTH)
    end_time = data.get("end_time", 10)
    
    # Define parameters
    parameters = {
        'num_steps': int(end_time/dt),
    }
    
    # Create model
    print("Creating model...")
    _model_instance = NetworkedModelNumba(
        dt=MINUTE_LENGTH,
        seed=42,
        parameters=parameters,
        verbose=True
    )

    return jsonify({
        "status": "ok",
    }), 201
    

@model_bp.route("/run", methods=["POST"])
def run_model():
    if _model_instance is None:
        return jsonify({"error": "model not initialized"}), 400
    
    while int(_model_instance.time / _model_instance.dt) < _model_instance.num_steps:
        _model_instance.step()
    
    stress = _model_instance.data["stress"].tolist()
    aversive_state = _model_instance.data["aversive_internal_state"].tolist()
    return jsonify({
        "stress": stress,
        "aversive_state": aversive_state,
    }), 200