# ML Skew

An end-to-end MLOps system for detecting **training-serving feature skew**, monitoring statistical data drift, tracking model experiments, serving predictions, and visualizing operational metrics.

The project uses the NYC Yellow Taxi dataset to train a fare-prediction model and demonstrates how small inconsistencies between training and production feature logic can materially change model predictions.

## Project objective

A model can perform well during training but fail in production even when the model weights have not changed.

Common causes include:

- Training uses miles while serving uses kilometres.
- Training and serving apply different timezone logic.
- Missing values are handled differently.
- Rush-hour rules change.
- Location mappings become inconsistent.
- Production feature distributions gradually change.

ML Skew detects these problems using two complementary approaches:

1. **Exact training-serving parity checks** for individual requests.
2. **Statistical drift detection** across batches of production data.

## Architecture

```text
NYC Yellow Taxi data
        ↓
Data validation and preparation
        ↓
Canonical feature engineering
        ↓
LightGBM model training
        ↓
Model evaluation and local artifacts
        ↓
MLflow experiment tracking
        ↓
MLflow Model Registry
        ↓
Champion model alias
        ↓
BentoML model import and REST serving
        ↓
Exact request-level skew detection
        ↓
NannyML batch-level drift detection
        ↓
Prometheus metrics
        ↓
Grafana dashboard
```

## Technology stack

| Area | Technology |
|---|---|
| Language | Python 3.13 |
| Monitoring environment | Python 3.12 |
| Dataset | NYC Yellow Taxi |
| Model | LightGBM regression |
| Validation and contracts | Pydantic |
| Experiment tracking | MLflow |
| Model Registry | MLflow Model Registry |
| Model serving | BentoML |
| Exact skew detection | Custom parity engine |
| Statistical drift | NannyML |
| Metrics | Prometheus |
| Dashboard | Grafana |
| Local infrastructure | Docker Compose |
| Testing | Pytest |
| Linting and formatting | Ruff |
| Static typing | mypy |

## Model features

The fare-regression model uses nine features:

```text
trip_distance_miles
passenger_count
pickup_location_id
dropoff_location_id
pickup_hour
pickup_day_of_week
pickup_month
is_weekend
is_rush_hour
```

The target column is:

```text
fare_amount
```

## Baseline model results

The baseline was trained using the first 100,000 January 2024 NYC Yellow Taxi records.

```text
Rows received:   100,000
Rows valid:       95,759
Training rows:    76,607
Validation rows:  19,152

MAE:              2.058969
RMSE:             6.705635
R²:               0.901455
```

### Metric interpretation

- **MAE** measures the average absolute prediction error.
- **RMSE** penalizes larger prediction errors more heavily.
- **R²** indicates how much fare variation is explained by the model.

An R² of approximately `0.901` means the model explains about 90.1% of validation-set fare variation.

## Training-serving skew

Training-serving skew occurs when the same raw input produces different engineered features in training and production.

Example:

```text
Raw trip distance: 4.5 miles

Offline training feature: 4.5
Online serving feature:   7.242048
```

The online path has incorrectly converted miles to kilometres.

The parity engine returns a structured mismatch:

```json
{
  "feature": "trip_distance_miles",
  "offline_value": 4.5,
  "online_value": 7.242048
}
```

## Supported skew simulations

The fault injector supports:

```text
none
distance_unit
timezone
missing_value
rush_hour_rule
location_mapping
```

Fault injection is used only for testing and demonstration.

## Statistical drift

Statistical drift is different from exact training-serving skew.

- Exact skew compares offline and online values for the same request.
- Statistical drift compares feature distributions across reference and analysis periods.

NannyML uses the Kolmogorov-Smirnov method for continuous features.

A real taxi distance-shift experiment produced:

```text
Feature:          trip_distance_miles
Method:           kolmogorov_smirnov
Threshold:        0.100000
Drift detected:   True
Alert count:      5 of 5 chunks
Maximum KS value: 0.303000
```

## Project structure

```text
ml-skew/
├── compose.observability.yml
├── observability/
│   ├── grafana/
│   │   ├── dashboards/
│   │   └── provisioning/
│   └── prometheus/
├── requirements-monitoring.txt
├── src/ml_skew/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── monitoring/
│   ├── observability/
│   ├── serving/
│   ├── tracking/
│   └── training/
└── tests/
    ├── data/
    ├── monitoring/
    ├── observability/
    ├── serving/
    ├── tracking/
    └── training/
```

## Main environment setup

The main application environment uses Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify the installation:

```bash
python -m pip check
pytest
ruff check src tests
ruff format --check src tests
python -m mypy src
```

## Dataset setup

Place the January 2024 NYC Yellow Taxi Parquet file at:

```text
data/raw/yellow_tripdata_2024-01.parquet
```

Dataset files and generated artifacts are excluded from Git.

## Start MLflow

Create the local MLflow runtime directory:

```bash
mkdir -p mlflow/artifacts
```

Start the MLflow tracking server:

```bash
mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --allowed-hosts "localhost:*,127.0.0.1:*" \
  --backend-store-uri sqlite:///mlflow/mlflow.db \
  --artifacts-destination ./mlflow/artifacts
```

Open:

```text
http://127.0.0.1:5000
```

## Train and log the baseline model

Keep MLflow running and execute:

```bash
python -m ml_skew.training.run_baseline \
  --dataset data/raw/yellow_tripdata_2024-01.parquet \
  --output-directory artifacts/baseline \
  --row-limit 100000 \
  --run-name baseline-real-taxi-2024-01
```

The command:

- Loads and validates the dataset.
- Builds canonical features.
- Trains the LightGBM model.
- Evaluates MAE, RMSE, and R².
- Saves local artifacts.
- Logs parameters, metrics, dataset lineage, and the model to MLflow.

Local artifacts are written to:

```text
artifacts/baseline/
├── model.joblib
├── metrics.json
└── metadata.json
```

## MLflow Model Registry

The registered model name is:

```text
ml-skew-fare-regressor
```

The serving-safe alias is:

```text
champion
```

Stable MLflow URI:

```text
models:/ml-skew-fare-regressor@champion
```

The alias can later be reassigned to a better model version without changing serving code.

## Import the MLflow model into BentoML

Keep MLflow running.

```bash
python - <<'PY'
import bentoml.mlflow
import mlflow

tracking_uri = "http://127.0.0.1:5000"
model_uri = "models:/ml-skew-fare-regressor@champion"

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)

model = bentoml.mlflow.import_model(
    name="ml-skew-fare-regressor",
    model_uri=model_uri,
    signatures={
        "predict": {
            "batchable": True,
            "batch_dim": 0,
        }
    },
    labels={
        "project": "ml-skew",
        "model-role": "champion",
    },
    metadata={
        "mlflow_model_uri": model_uri,
        "task": "taxi-fare-regression",
    },
)

print(model.tag)
PY
```

Save the printed BentoML tag. Example:

```text
ml-skew-fare-regressor:ybaxmkurs6mumjrr
```

## Start the BentoML API

Set the imported model tag:

```bash
export BENTO_MODEL_TAG="ml-skew-fare-regressor:<version>"
```

Start the service:

```bash
bentoml serve \
  ml_skew.serving.service:FarePredictionService \
  --host 0.0.0.0 \
  --port 3000
```

Open the Swagger interface:

```text
http://127.0.0.1:3000
```

## REST endpoints

### Engineered-feature prediction

```text
POST /predict
```

Example request:

```bash
curl --request POST \
  --url http://127.0.0.1:3000/predict \
  --header 'Content-Type: application/json' \
  --data '{
    "trip_distance_miles": 4.5,
    "passenger_count": 2,
    "pickup_location_id": 132,
    "dropoff_location_id": 236,
    "pickup_hour": 8,
    "pickup_day_of_week": 1,
    "pickup_month": 1,
    "is_weekend": 0,
    "is_rush_hour": 1
  }'
```

### Raw monitored prediction

```text
POST /predict-raw
```

Clean request:

```bash
curl --request POST \
  --url http://127.0.0.1:3000/predict-raw \
  --header 'Content-Type: application/json' \
  --data '{
    "pickup_datetime": "2024-01-08T08:30:00Z",
    "pickup_location_id": 132,
    "dropoff_location_id": 236,
    "passenger_count": 2,
    "trip_distance_miles": 4.5,
    "skew_mode": "none"
  }'
```

Example response:

```json
{
  "predicted_fare_amount": 23.096771125349996,
  "model_tag": "ml-skew-fare-regressor:ybaxmkurs6mumjrr",
  "skew": {
    "detected": false,
    "skew_mode": "none",
    "mismatch_count": 0,
    "mismatches": []
  }
}
```

Distance-unit skew request:

```bash
curl --request POST \
  --url http://127.0.0.1:3000/predict-raw \
  --header 'Content-Type: application/json' \
  --data '{
    "pickup_datetime": "2024-01-08T08:30:00Z",
    "pickup_location_id": 132,
    "dropoff_location_id": 236,
    "passenger_count": 2,
    "trip_distance_miles": 4.5,
    "skew_mode": "distance_unit"
  }'
```

Example skewed response:

```json
{
  "predicted_fare_amount": 31.631186176110475,
  "model_tag": "ml-skew-fare-regressor:ybaxmkurs6mumjrr",
  "skew": {
    "detected": true,
    "skew_mode": "distance_unit",
    "mismatch_count": 1,
    "mismatches": [
      {
        "feature": "trip_distance_miles",
        "offline_value": 4.5,
        "online_value": 7.242048
      }
    ]
  }
}
```

The transformation error changes the prediction from approximately `23.10` to `31.63`.

## Monitoring environment setup

NannyML 0.13.1 requires Python below 3.13, so it runs in an isolated Python 3.12 environment.

```bash
python3.12 -m venv .venv-monitoring
source .venv-monitoring/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-monitoring.txt
python -m pip check
```

On macOS, XGBoost also requires OpenMP:

```bash
brew install libomp
```

## Run statistical drift analysis

Activate the monitoring environment:

```bash
source .venv-monitoring/bin/activate
```

Run:

```bash
PYTHONPATH=src python -m ml_skew.monitoring.run_drift_analysis \
  --dataset data/raw/yellow_tripdata_2024-01.parquet \
  --output artifacts/monitoring/drift-report.json \
  --row-limit 2000 \
  --reference-rows 1000 \
  --analysis-rows 500 \
  --shifted-feature trip_distance_miles \
  --shift-multiplier 1.609344 \
  --chunk-number 5 \
  --upper-threshold 0.1
```

The JSON report is written to:

```text
artifacts/monitoring/drift-report.json
```

## Prometheus metrics

BentoML exposes metrics at:

```text
http://127.0.0.1:3000/metrics
```

Custom project metrics:

```text
ml_skew_training_serving_skew_detections_total
ml_skew_latest_predicted_fare_amount
```

BentoML also exposes standard HTTP request, latency, and error metrics.

## Start Prometheus and Grafana

Keep BentoML running on `0.0.0.0:3000`.

Start the observability stack:

```bash
docker compose \
  -f compose.observability.yml \
  up -d
```

Check status:

```bash
docker compose \
  -f compose.observability.yml \
  ps
```

Open:

```text
Prometheus: http://127.0.0.1:9090
Grafana:    http://127.0.0.1:3001
```

Default Grafana credentials:

```text
Username: admin
Password: admin
```

Provisioned dashboard:

```text
Dashboards → ML Skew → ML Skew Monitoring
```

The dashboard includes:

- BentoML service availability
- Total skew detections
- Latest predicted fare
- Skew detections by mode

Stop the observability stack:

```bash
docker compose \
  -f compose.observability.yml \
  down
```

## Testing strategy

The project includes tests for:

- Data loading and validation
- Feature preparation
- Offline and online feature adapters
- Exact feature parity
- Fault injection
- LightGBM training
- Artifact persistence
- MLflow tracking
- Model Registry alias promotion
- BentoML request validation
- BentoML prediction logic
- Runtime skew reports
- Drift dataset preparation
- NannyML drift calculation
- Prometheus metric updates

Run the main test suite:

```bash
source .venv/bin/activate
pytest
```

Run real NannyML integration tests:

```bash
source .venv-monitoring/bin/activate

PYTHONPATH=src python -m unittest discover \
  -s tests/monitoring \
  -p 'test_nannyml_drift.py' \
  -v
```

The main environment skips NannyML integration tests because NannyML is intentionally isolated in Python 3.12.

## Quality checks

```bash
ruff check src tests
ruff format --check src tests
python -m mypy src
python -m pip check
```

Current validated quality result:

```text
87 tests passed
2 expected NannyML skips
Ruff passed
Formatting passed
mypy passed
Dependency check passed
Grafana JSON valid
Docker Compose valid
```

## Key engineering decisions

### Separate offline and online adapters

Training and production often use different execution paths or release cycles.

Separate adapters make it possible to verify that both paths still generate identical features.

### Canonical feature logic

A canonical transformation defines the expected feature behaviour and reduces accidental differences.

### Exact skew plus statistical drift

Exact parity detects deterministic request-level mismatches.

NannyML detects population-level distribution changes.

Both are required because they solve different monitoring problems.

### MLflow plus BentoML

MLflow manages experiments, lineage, model versions, and promotion.

BentoML packages and serves the selected model through a production-style API.

### Separate NannyML environment

The main application uses Python 3.13, while NannyML 0.13.1 requires Python below 3.13.

A separate Python 3.12 environment avoids dependency conflicts.

### Fault injection

Intentional skew modes create repeatable demonstrations and prove that monitoring works before a real incident occurs.

## Interview summary

> I built an end-to-end MLOps system that trains a LightGBM taxi-fare model, tracks experiments and model versions with MLflow, serves the champion model through BentoML, detects exact training-serving feature mismatches for individual requests, and performs batch-level statistical drift detection using NannyML. I also exposed operational metrics through Prometheus and created a provisioned Grafana monitoring dashboard.

## Current limitations

- Prometheus custom metrics are stored in process memory and reset when BentoML restarts.
- The demonstration uses a local SQLite MLflow backend.
- NannyML runs as a separate analysis process rather than a scheduled production job.
- Model retraining is not automatically triggered.
- The project does not include a custom React or Streamlit frontend.
- The Docker Compose stack currently covers Prometheus and Grafana; BentoML and MLflow run on the host.

## Future enhancements

- Containerize MLflow and BentoML.
- Add PostgreSQL for MLflow metadata.
- Persist Prometheus metrics across service restarts.
- Add Grafana alert rules.
- Add scheduled drift analysis.
- Add estimated model-performance monitoring.
- Automate retraining and champion promotion.
- Add a lightweight frontend for demonstrations.
- Deploy the complete stack to a cloud environment.

## License

This repository is intended for educational, portfolio, and MLOps demonstration purposes.
