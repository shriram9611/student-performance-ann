"""Serving tests. They use a tiny stand-in model, so TensorFlow is NOT needed to run them.

Run:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import api
import app as webapp
from common import FEATURES, load_data, prepare_data
from predictor import Predictor

GOOD = {"age": 20, "attendance_pct": 82, "study_hours_per_week": 22, "previous_score": 71,
        "assignment_completion_pct": 80, "internal_assessment_score": 68,
        "participation_score": 7, "sleep_hours_per_day": 7,
        "extracurricular_hours_per_week": 4, "failed_subjects": 0}


class FakeModel:
    def predict(self, x, verbose=0):
        return np.array([[0.05, 0.15, 0.80]])


def make_predictor():
    scaler = StandardScaler().fit(np.random.rand(20, len(FEATURES)))
    return Predictor(FakeModel(), scaler)


class ServingTests(unittest.TestCase):
    def test_api_predict_ok(self):
        c = api.create_app(make_predictor()).test_client()
        r = c.post("/predict", json=GOOD)
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["predicted_performance"], "High")
        self.assertAlmostEqual(body["confidence"], 0.8)

    def test_api_missing_feature(self):
        c = api.create_app(make_predictor()).test_client()
        bad = dict(GOOD)
        bad.pop("age")
        self.assertEqual(c.post("/predict", json=bad).status_code, 400)

    def test_api_out_of_range(self):
        c = api.create_app(make_predictor()).test_client()
        self.assertEqual(c.post("/predict", json={**GOOD, "attendance_pct": 140}).status_code, 400)

    def test_api_not_json(self):
        c = api.create_app(make_predictor()).test_client()
        self.assertEqual(c.post("/predict", data="x").status_code, 400)

    def test_web_app(self):
        c = webapp.create_app(make_predictor()).test_client()
        self.assertEqual(c.get("/").status_code, 200)
        r = c.post("/", data={k: str(v) for k, v in GOOD.items()})
        self.assertIn(b"High", r.data)
        self.assertIn(b"80.0%", r.data)

    def test_preprocessing_no_leakage_shapes(self):
        d = prepare_data(load_data())
        self.assertEqual(d["X_train"].shape[1], len(FEATURES))
        self.assertEqual(len(d["y_train"]) + len(d["y_val"]) + len(d["y_test"]), 15000)
        self.assertAlmostEqual(float(d["X_train"].mean()), 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
