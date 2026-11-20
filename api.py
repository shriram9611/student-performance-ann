"""JSON prediction API.

Run:   python api.py            (serves on http://127.0.0.1:5000)
Test:  curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
         -d '{"age":20,"attendance_pct":82,"study_hours_per_week":22,"previous_score":71,
              "assignment_completion_pct":80,"internal_assessment_score":68,
              "participation_score":7,"sleep_hours_per_day":7,
              "extracurricular_hours_per_week":4,"failed_subjects":0}'
"""
from flask import Flask, jsonify, request

from common import CLASSES, FEATURE_SPECS
from predictor import Predictor, ValidationError


def create_app(predictor=None):
    app = Flask(__name__)
    holder = {"p": predictor}

    def get_predictor():
        if holder["p"] is None:
            holder["p"] = Predictor.from_disk()      # load once, on first request
        return holder["p"]

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.get("/features")
    def features():
        return jsonify(features=FEATURE_SPECS, classes=CLASSES)

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify(error="Send a JSON body with Content-Type: application/json."), 400
        try:
            return jsonify(get_predictor().predict(payload))
        except ValidationError as e:
            return jsonify(error=str(e)), 400
        except FileNotFoundError as e:
            return jsonify(error=str(e)), 503

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000)
