# AI-Based Student Performance Risk Predictor (ANN)

A feed-forward Artificial Neural Network (Keras / TensorFlow) that predicts a student's performance category - **Low, Medium or High** - from academic and behavioural data. The project includes a logistic-regression baseline, evaluation, a saved model, a web app and a JSON API.

## Dataset
`data/student_data.csv` - 15,000 students, no missing values, no duplicates, balanced classes (Low 4950, Medium 5100, High 4950).

| Role | Columns |
|---|---|
| Input features (10) | age, attendance_pct, study_hours_per_week, previous_score, assignment_completion_pct, internal_assessment_score, participation_score, sleep_hours_per_day, extracurricular_hours_per_week, failed_subjects |
| Target | performance_category (Low / Medium / High) |
| Dropped | student_id (identifier only) |

## Project structure
```
student-performance-ann/
├── data/student_data.csv
├── notebooks/ann_training.ipynb     # analysis, baseline, ANN, evaluation (step by step)
├── models/                          # created by train.py (student_performance.keras, scaler.joblib, metadata.json)
├── reports/                         # created by train.py (plots + evaluation_report.md)
├── tests/test_serving.py            # API / app / preprocessing tests (no TensorFlow needed)
├── common.py                        # constants, data loading, preprocessing, metrics
├── train.py                         # trains baseline + ANN, evaluates, saves everything
├── predictor.py                     # loads saved model and predicts
├── app.py                           # web application (port 8000)
├── api.py                           # JSON API (port 5000)
├── requirements.txt
└── README.md
```

## Setup
Python 3.10 - 3.13 (use a version supported by your TensorFlow release).
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
1. **Train** (creates the model and the evaluation report):
   ```bash
   python train.py
   ```
   or run `notebooks/ann_training.ipynb` top to bottom (`jupyter notebook`).
2. **Web app:** `python app.py` then open http://127.0.0.1:8000, enter the student's details and press *Predict*.
3. **API:** `python api.py`, then
   ```bash
   curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{
     "age": 20, "attendance_pct": 82, "study_hours_per_week": 22, "previous_score": 71,
     "assignment_completion_pct": 80, "internal_assessment_score": 68,
     "participation_score": 7, "sleep_hours_per_day": 7,
     "extracurricular_hours_per_week": 4, "failed_subjects": 0}'
   ```
   Response:
   ```json
   {"predicted_performance": "High", "confidence": 0.89, "probabilities": {"Low": 0.01, "Medium": 0.10, "High": 0.89}}
   ```
   Other endpoints: `GET /health`, `GET /features`. Missing or out-of-range values return HTTP 400 with an error message.
4. **Tests:** `python -m unittest discover -s tests -v`

## Method
* **Preprocessing:** labels encoded Low=0 / Medium=1 / High=2; stratified 70/15/15 train/validation/test split; `StandardScaler` fitted on the training split only (no leakage). Scaling matters because inputs have very different ranges and unscaled features slow and destabilise gradient descent.
* **Baseline:** Logistic Regression on the same scaled data.
* **ANN:** `Input(10) -> Dense(32, ReLU) -> Dropout(0.2) -> Dense(16, ReLU) -> Dense(3, Softmax)`. The data is small and tabular, so a compact network is enough; two hidden layers capture feature interactions, dropout limits overfitting, softmax gives class probabilities.
* **Training:** Adam (lr 1e-3), sparse categorical cross-entropy, batch size 64, up to 150 epochs with early stopping and learning-rate reduction on validation loss.
* **Evaluation:** accuracy, macro precision / recall / F1 and confusion matrix on the unseen test split - see `reports/evaluation_report.md` after training.

## Key concepts (short answers)
* **Perceptron vs multi-layer ANN:** a perceptron is a single neuron that can only separate classes with a straight line; an ANN stacks layers with non-linear activations and learns complex boundaries.
* **Neurons, weights, biases:** a neuron computes `activation(w·x + b)`; weights set each input's importance, bias shifts the threshold.
* **Activation functions:** ReLU in hidden layers adds non-linearity cheaply; softmax in the output layer converts 3 scores into probabilities that sum to 1.
* **Forward propagation:** input -> hidden layer 1 -> hidden layer 2 -> softmax probabilities; the largest probability is the prediction and confidence.
* **Backpropagation:** the loss measures the error; the chain rule passes it backwards to get each weight's gradient; the optimizer updates weights against the gradient.
* **Loss and optimizer:** cross-entropy penalises confident wrong predictions; Adam adapts the step size per weight.
* **Inference:** `train.py` saves the model (`.keras`) and the scaler; `predictor.py` loads both, scales the new student's data the same way and returns the probabilities.

## Notes
* Real-world accuracy is limited by the data: the categories overlap, so expect roughly 70 % accuracy, with the ANN performing close to the logistic-regression baseline. That is a legitimate finding to discuss in the report/demo.
* Scope follows the assignment: no CNNs, cloud deployment or authentication.
