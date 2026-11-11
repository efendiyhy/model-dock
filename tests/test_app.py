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
