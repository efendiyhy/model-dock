"""Run with python -m pytest tests -q (pytest and httpx needed)."""
import io
import json
import os
import tempfile
import uuid
from pathlib import Path

# Set before importing server module so tests cannot write into real demo state.
_TEST_DATA = tempfile.TemporaryDirectory(prefix="modeldock-tests-")
os.environ["MODELDOCK_DATA_DIR"] = _TEST_DATA.name

import joblib
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from app.main import app

client = TestClient(app)
# Existing regression tests use one authenticated local admin session.
admin_login = client.post('/api/auth/login', json={'username': 'ekinstru', 'password': 'sandikuat'})
assert admin_login.status_code == 200
_csrf = admin_login.json()['csrf_token']
_original_request = client.request

def _request_with_csrf(method, url, **kwargs):
    if url.startswith('/api/') and method.upper() in {'POST', 'PUT', 'PATCH', 'DELETE'}:
        headers = dict(kwargs.get('headers') or {})
        headers.setdefault('X-CSRF-Token', _csrf)
        kwargs['headers'] = headers
    return _original_request(method, url, **kwargs)

client.request = _request_with_csrf


def key():
    return client.get('/api/key').json()['api_key']


def test_home_sample_and_fastapi_docs():
    assert client.get('/').status_code == 200
    assert 'ModelDock' in client.get('/').text
    assert client.get('/static/app.js').status_code == 200
    assert client.get('/docs').status_code == 200
    assert client.get('/health').json()['status'] == 'ok'
    overview = client.get('/api/overview').json()
    assert overview['running'] >= 1
    assert any(x['slug'] == 'iris-demo' for x in client.get('/api/models').json())


def test_authenticated_prediction_and_logging():
    payload = {"inputs": {"sepal length (cm)":5.1, "sepal width (cm)":3.5, "petal length (cm)":1.4, "petal width (cm)":0.2}}
    assert client.post('/v1/iris-demo/predict', json=payload).status_code == 401
    response = client.post('/v1/iris-demo/predict', json=payload, headers={"X-API-Key": key()})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['predictions'] == [0]
    assert result['rows'] == 1
    assert len(result['probabilities'][0]) == 3
    batch = client.post('/v1/iris-demo/predict',json={"inputs":[payload['inputs'],payload['inputs']]},headers={"X-API-Key":key()})
    assert batch.status_code == 200 and len(batch.json()['predictions']) == 2
    assert client.post('/v1/iris-demo/predict',json={"inputs": {"wrong":1}},headers={"X-API-Key":key()}).status_code == 422
    assert any(log['model_slug']=='iris-demo' for log in client.get('/api/requests').json())
    assert len(client.get('/v1/iris-demo/schema', headers={"X-API-Key":key()}).json()['features']) == 4


def test_upload_predict_stop_resume_delete_and_key_rotation():
    model = LinearRegression().fit([[1,2], [2,3], [3,4], [4,5]], [3,5,7,9])
    buf = io.BytesIO()
    joblib.dump(model, buf)
    slug = 'test-model-'+uuid.uuid4().hex[:8]
    file = ("trained_model.joblib", buf.getvalue(), "application/octet-stream")
    untrusted = client.post('/api/models', files={"file":file}, data={"name":slug,"features":json.dumps(['x1','x2'])})
    assert untrusted.status_code == 400
    created = client.post('/api/models',files={"file":file},data={"name":slug,"features":json.dumps(['x1','x2']),"trusted":"true"})
    assert created.status_code == 201, created.text
    assert created.json()['model']['task'] == 'regression'
    result = client.post(f'/v1/{slug}/predict',headers={'X-API-Key':key()},json={"inputs":{"x1":2,"x2":3}})
    assert result.status_code == 200, result.text
    assert abs(result.json()['predictions'][0]-5)<.01
    pause = client.patch(f'/api/models/{slug}/status',json={'active':False})
    assert pause.status_code == 200
    assert client.post(f'/v1/{slug}/predict',headers={'X-API-Key':key()},json={"inputs":{"x1":2,"x2":3}}).status_code == 503
    assert client.patch(f'/api/models/{slug}/status',json={'active':True}).status_code == 200
    assert client.post(f'/v1/{slug}/predict',headers={'X-API-Key':key()},json={"inputs":[2,3]}).status_code == 200
    previous = key()
    new = client.post('/api/key/regenerate').json()['api_key']
    assert previous != new
    assert client.post(f'/v1/{slug}/predict',headers={'X-API-Key':previous},json={"inputs":[2,3]}).status_code == 401
    assert client.post(f'/v1/{slug}/predict',headers={'X-API-Key':new},json={"inputs":[2,3]}).status_code == 200
    assert client.delete(f'/api/models/{slug}').status_code == 200
    assert client.get(f'/api/models/{slug}').status_code == 404


def test_invalid_file_and_input_schema():
    fake = client.post('/api/models',files={"file":('fake.txt',b'abc')},data={'name':'fake','trusted':'true'})
    assert fake.status_code == 400
    assert client.post('/api/models/iris-demo/test',json={'inputs':{}}).status_code == 422
    assert client.post('/api/models/iris-demo/test',json={'inputs':[1,2]}).status_code == 422


def test_upload_custom_class_model_ncs_models_gabungan():
    """Models pickled as ncs_models.Gabungan (SVM ensemble) must upload and predict."""
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from ncs_models import Gabungan

    feats = ['f_a', 'f_b', 'f_c']
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = np.array(['Absent', 'Mild', 'Moderate'])[np.arange(60) % 3]
    X[y == 'Moderate', 0] += 4
    import pandas as pd
    pipe = Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler()),
                     ('m', SVC(probability=True, random_state=42))]).fit(pd.DataFrame(X, columns=feats), y)
    model = Gabungan()
    model.__dict__.update({'anggota': [(feats, None, False, 1)], 'seed': 42,
                           'classes_': np.array(['Absent', 'Mild', 'Moderate']),
                           'prior_': np.array([1/3, 1/3, 1/3]), 'model_': [(feats, pipe, False, 1)]})
    buf = io.BytesIO()
    joblib.dump(model, buf)
    slug = 'svm-gabungan-' + uuid.uuid4().hex[:8]
    created = client.post('/api/models', files={"file": ("SVM_Dasar.joblib", buf.getvalue())},
                          data={"name": slug, "trusted": "true"})
    assert created.status_code == 201, created.text
    meta = created.json()['model']
    assert meta['features'] == feats and meta['task'] == 'classification'
    result = client.post(f'/v1/{slug}/predict', headers={'X-API-Key': key()},
                         json={"inputs": [{"f_a": 5, "f_b": 0, "f_c": 0}, {"f_a": None, "f_b": 0, "f_c": 0}]})
    assert result.status_code == 200, result.text
    body = result.json()
    assert body['predictions'][0] == 'Moderate'
    assert all(abs(sum(p) - 1) < 1e-6 for p in body['probabilities'])
    assert client.post(f'/v1/{slug}/predict', headers={'X-API-Key': key()}, json={"inputs": [5, 0, 0]}).status_code == 200
    assert client.delete(f'/api/models/{slug}').status_code == 200


def test_missing_custom_module_gives_clear_error():
    # Hand-craft a pickle that references a module that does not exist on the server.
    raw = b'\x80\x04\x8c\x0eno_such_module\x8c\x03Foo\x93)\x81.'
    created = client.post('/api/models', files={"file": ("x.pkl", raw)},
                          data={"name": 'missing-' + uuid.uuid4().hex[:6], "trusted": "true"})
    assert created.status_code == 422
    assert 'custom_modules' in created.json()['detail']


def _pickle_with_custom_module(module_name, code):
    """Build a joblib file whose class lives in a module the server does not have yet."""
    import importlib.util
    import sys
    tmp = Path(tempfile.mkdtemp()) / f"{module_name}.py"
    tmp.write_text(code)
    spec = importlib.util.spec_from_file_location(module_name, tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    buf = io.BytesIO()
    joblib.dump(mod.Doubler(), buf)
    del sys.modules[module_name]
    return buf.getvalue(), code.encode()


DOUBLER = '''
import numpy as np
class Doubler:
    feature_names_in_ = np.array(["x"], dtype=object)
    n_features_in_ = 1
    def predict(self, X):
        return np.asarray(X, dtype=float)[:, 0] * 2
'''


def test_upload_model_together_with_custom_code_file():
    module = 'custom_' + uuid.uuid4().hex[:8]
    model_bytes, code = _pickle_with_custom_module(module, DOUBLER)
    slug = 'doubler-' + uuid.uuid4().hex[:6]
    # Without the .py file the upload fails with a helpful message.
    missing = client.post('/api/models', files={"file": ("m.joblib", model_bytes)}, data={"name": slug, "trusted": "true"})
    assert missing.status_code == 422 and 'File kode tambahan' in missing.json()['detail']
    created = client.post('/api/models', data={"name": slug, "trusted": "true"},
                          files=[("file", ("m.joblib", model_bytes)), ("code_files", (f"{module}.py", code))])
    assert created.status_code == 201, created.text
    assert created.json()['code_files'] == [f"{module}.py"]
    assert f"{module}.py" in client.get('/api/custom-modules').json()['uploaded']
    result = client.post(f'/v1/{slug}/predict', headers={'X-API-Key': key()}, json={"inputs": {"x": 21}})
    assert result.status_code == 200 and result.json()['predictions'] == [42.0]
    assert client.delete(f'/api/models/{slug}').status_code == 200


def test_invalid_code_files_are_rejected_and_rolled_back():
    model_bytes, _ = _pickle_with_custom_module('custom_' + uuid.uuid4().hex[:8], DOUBLER)
    name = 'bad-' + uuid.uuid4().hex[:6]
    for fname, body in [("bad.py", b"def oops(:\n"), ("json.py", b"x=1\n"), ("notes.txt", b"x=1\n")]:
        r = client.post('/api/models', data={"name": name, "trusted": "true"},
                        files=[("file", ("m.joblib", model_bytes)), ("code_files", (fname, body))])
        assert r.status_code == 400, (fname, r.text)
    # Valid code but the model still can't load (wrong module) -> code file is removed again.
    stray = 'stray_' + uuid.uuid4().hex[:6]
    r = client.post('/api/models', data={"name": name, "trusted": "true"},
                    files=[("file", ("m.joblib", model_bytes)), ("code_files", (f"{stray}.py", b"x = 1\n"))])
    assert r.status_code == 422
    assert f"{stray}.py" not in client.get('/api/custom-modules').json()['uploaded']
