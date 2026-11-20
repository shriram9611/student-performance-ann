"""Simple web application: enter a student's details, get the ANN prediction.

Run:  python app.py      then open http://127.0.0.1:8000
"""
from flask import Flask, render_template_string, request

from common import CLASSES, FEATURE_SPECS
from predictor import Predictor, ValidationError

PAGE = """
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Student Performance Predictor</title>
<style>
 body{font-family:system-ui,sans-serif;background:#f4f6fb;margin:0;color:#1d2433}
 main{max-width:760px;margin:32px auto;padding:0 16px}
 h1{font-size:1.5rem;margin-bottom:4px} p.sub{color:#5b6477;margin-top:0}
 form{background:#fff;border-radius:12px;padding:20px;display:grid;grid-template-columns:1fr 1fr;gap:14px;box-shadow:0 1px 4px #0001}
 label{display:flex;flex-direction:column;font-size:.85rem;gap:4px;color:#3a4256}
 input{padding:8px;border:1px solid #c9d0de;border-radius:8px;font-size:1rem}
 button{grid-column:1/-1;padding:12px;border:0;border-radius:8px;background:#2f5bea;color:#fff;font-size:1rem;cursor:pointer}
 .card{background:#fff;border-radius:12px;padding:20px;margin-top:20px;box-shadow:0 1px 4px #0001}
 .err{background:#fdecec;color:#a12626}
 .big{font-size:1.6rem;font-weight:700}
 .bar{background:#e6eaf3;border-radius:6px;height:14px;margin:4px 0 10px;overflow:hidden}
 .bar span{display:block;height:100%;background:#2f5bea}
 @media(max-width:560px){form{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Student Performance Risk Predictor</h1>
<p class="sub">Feed-forward ANN (Keras) &middot; classes: {{ classes|join(' / ') }}</p>
<form method="post">
 {% for f in specs %}
 <label>{{ f.label }}
  <input type="number" name="{{ f.name }}" min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}"
         value="{{ values[f.name] }}" required></label>
 {% endfor %}
 <button type="submit">Predict</button>
</form>
{% if error %}<div class="card err">{{ error }}</div>{% endif %}
{% if result %}
<div class="card">
 <div>Predicted performance</div>
 <div class="big">{{ result.predicted_performance }}</div>
 <div>Prediction probability: <b>{{ '%.1f'|format(result.confidence*100) }}%</b></div>
 <hr style="border:0;border-top:1px solid #e6eaf3;margin:14px 0">
 {% for c, p in result.probabilities.items() %}
  <div>{{ c }} &mdash; {{ '%.1f'|format(p*100) }}%</div>
  <div class="bar"><span style="width:{{ p*100 }}%"></span></div>
 {% endfor %}
</div>
{% endif %}
</main></body></html>
"""


def create_app(predictor=None):
    app = Flask(__name__)
    holder = {"p": predictor}

    @app.route("/", methods=["GET", "POST"])
    def index():
        values = {f["name"]: f["default"] for f in FEATURE_SPECS}
        result = error = None
        if request.method == "POST":
            values = {f["name"]: request.form.get(f["name"], "") for f in FEATURE_SPECS}
            try:
                if holder["p"] is None:
                    holder["p"] = Predictor.from_disk()
                result = holder["p"].predict(values)
            except (ValidationError, FileNotFoundError) as e:
                error = str(e)
        return render_template_string(PAGE, specs=FEATURE_SPECS, classes=CLASSES,
                                      values=values, result=result, error=error)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=8000)
