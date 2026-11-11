# ModelDock — Local ML Inference Hosting Prototype

**Upload a pre-trained machine learning model → automatically receive a FastAPI endpoint → make actual predictions from another application.** This is a functional *single-operator LOCAL demo*, not a cloud service ready for strangers.

## Features

- Professional responsive dark dashboard, model registry, API playground, API keys, request logs, and documentation.
- Upload **trusted** `.joblib`, `.pkl`, `.pickle` trained models (scikit-learn pipelines/estimators, and compatible estimators with `.predict()`).
- Detect `feature_names_in_` and `n_features_in_`; optionally enter JSON array of ordered feature names.
- Per-model prediction URL: `POST /v1/{model-slug}/predict` and model schema `GET /v1/{model-slug}/schema`.
- Predictions (single or batch), classification probabilities when available, start/stop, delete, one demo API key, real SQLite request logs.
- Real pre-trained Iris model included and loaded on first run.
- FastAPI interactive Swagger docs at `/docs`.
- No training screen or ML fitting in the platform itself.

## Windows (easiest)

1. Install **Python 3.10 or newer** (enable "Add python.exe to PATH").
2. Extract the ZIP, ensure internet for the initial Python package install.
3. Double-click **`start.bat`** and allow first-time dependencies to install.
4. Open **http://127.0.0.1:8000** (should launch automatically). Don't close the terminal while testing.

## Linux / macOS

```bash
unzip modeldock-demo.zip
cd modeldock-demo
./start.sh
```

Then open http://127.0.0.1:8000.

## Optional Docker local test

```bash
docker compose up --build
```

Then open http://127.0.0.1:8000. Docker port is **bound to localhost**.

## Test the built-in model (no uploads needed)

1. Open **API Playground**.
2. Select **Iris Species Classifier** (auto-installed on first run).
3. The sample JSON input appears automatically. Click **Run Prediction**.
4. Observe real `predictions` and `probabilities` from the pre-trained scikit-learn pipeline.
5. Open **API Keys** and copy the demo API key to call the endpoint externally.

### Call FastAPI from another backend

```python
import requests

key = "COPY_YOUR_KEY_FROM_DASHBOARD"
response = requests.post(
    "http://127.0.0.1:8000/v1/iris-demo/predict",
    headers={"X-API-Key": key},
    json={"inputs": {
      "sepal length (cm)": 5.1,
      "sepal width (cm)": 3.5,
      "petal length (cm)": 1.4,
      "petal width (cm)": 0.2
    }},
    timeout=30,
)
print(response.status_code, response.json())
```

Test without installing `requests` using cURL (the dashboard API Keys page also provides a copyable example):

```bash
curl -X POST http://127.0.0.1:8000/v1/iris-demo/predict \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: YOUR_KEY' \
  -d '{"inputs":{"sepal length (cm)":5.1,"sepal width (cm)":3.5,"petal length (cm)":1.4,"petal width (cm)":0.2}}'
```

### Upload your own trained model

Save an already-trained scikit-learn estimator/pipeline:

```python
import joblib
joblib.dump(trained_pipeline, 'my_model.joblib')
```

Use **Deploy New Model** to upload it, tick the trust acknowledgement, and click Deploy. For estimators trained on unnamed numpy arrays, ModelDock generates `feature_1`, `feature_2`, etc. If naming manually, input e.g. `["temperature", "humidity", "pm25"]` **in exactly the order used during training**.

**Tip:** Save the complete preprocessing pipeline along with the estimator (for example `StandardScaler` + classifier) rather than the final estimator alone. The demo does not support automatically installing custom model dependencies.

## Models with custom classes (e.g. `ncs_models.Gabungan`)

A joblib/pickle file only stores an object's *data*; the Python **class** must be importable on the server
under the same module name. If upload fails with `No module named '...'`, copy the `.py` file that defines
the class (the one imported in your training notebook) in the **"File kode tambahan"** field of the upload
dialog, together with the model (several `.py` files are allowed, max. 1 MB each). Uploaded files are stored
in `data/custom_modules/` and take priority over the bundled ones. You can also copy the file into the
**`custom_modules/`** folder manually and restart (override that folder with `MODELDOCK_CUSTOM_MODULES_DIR`).

`custom_modules/ncs_models.py` is already included: it supports `Gabungan` ensemble models (e.g. `SVM_Dasar.joblib`,
SimpleImputer → StandardScaler → SVC). Feature names and classes are detected automatically, and prediction is a weighted
average of each member's `predict_proba`. If you have the original `ncs_models.py`, you can replace this file with it.

Classes defined directly in a notebook (`__main__`) cannot be loaded from a server. Move the class into a `.py` file,
import it from there, then retrain/re-save the model.

## Structure

```
app/main.py              FastAPI backend / upload / model registry / prediction
app/static/index.html    Dashboard UI shell
app/static/style.css     Responsive dark styling
app/static/app.js        Dashboard navigation, data, playground, actions
examples/iris_classifier.joblib  Pre-trained Iris sample
examples/create_sample.py       Optional example for recreating sample artifact
requirements.txt        Python packages
start.bat / start.sh     Local launchers
Dockerfile / compose.yaml   Optional local Docker
```

Runtime state is persisted in `data/` (auto-created on first start): SQLite DB, model files, and `demo_api_key.txt`. Deleting the folder resets the demo.

## Known limits / security (read before VPS)

This is deliberately **NOT safe to expose to the public internet**:

1. **There is no admin login.** Admin routes can list models, upload/delete files, and display/regenerate the API key. Run only on trusted localhost.
2. **Pickle/joblib deserialization can run arbitrary code.** The checkbox is NOT a security barrier; never upload models from strangers or arbitrary users. Multi-user public hosting requires sandboxed/containerized untrusted inference workers, ideally safer model formats and artifact scanning; do not run untrusted models in the API/web server process.
3. **Models share one Python process**, with no per-model environment, CPU/memory limits, or queues. Some models need specific scikit-learn versions or packages and may be incompatible. A production system needs separate isolated workers and environment pinning.
4. One single demo key is kept in a local plaintext file to allow dashboard testing. Production needs proper login, stored hashed keys, scoped keys, quotas, audit logs, TLS, and secure key lifecycle.
5. CORS only allows a short local development origin list by default. For a separate local web frontend, configure `MODELDOCK_ALLOWED_ORIGINS` comma-separated; avoid `*` for real client apps. Keep secret keys in server-side applications, **not** public web JavaScript.
6. This demo supports **scikit-learn compatible pickled estimators** only, not PyTorch, TensorFlow or ONNX yet. Maximum upload 35MB and batch 100 rows.
7. All calculations happen locally on your CPU. There is no connection to Hugging Face, OpenAI, or third-party servers.

For a real VPS product, the next iteration should include an authenticated frontend, DB users/projects/API keys, upload scanning, secure Docker worker isolation (no privileged Docker socket exposed), a job/worker queue, limits, Nginx/Caddy + HTTPS and backups.
