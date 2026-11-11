"""New admin-auth regression tests for ModelDock v0.2."""
import io
import os
import tempfile
from pathlib import Path

TEST_DIR = tempfile.TemporaryDirectory(prefix="modeldock-auth-")
os.environ["MODELDOCK_DATA_DIR"] = TEST_DIR.name
import joblib
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from app.main import app


def login(c):
    return c.post('/api/auth/login', json={'username': 'ekinstru', 'password': 'sandikuat'})


def test_login_and_protected_admin_endpoints():
    with TestClient(app) as c:
        assert c.get('/').status_code == 200
        assert c.get('/api/models').status_code == 401
        assert c.get('/api/key').status_code == 401
        assert c.get('/api/overview').status_code == 401
        assert c.get('/docs').status_code == 401
        assert c.post('/api/auth/login', json={'username':'ekinstru','password':'bad'}).status_code == 401
        r = login(c)
        assert r.status_code == 200, r.text
        csrf = r.json()['csrf_token']
        assert 'httponly' in r.headers['set-cookie'].lower()
        assert 'samesite=strict' in r.headers['set-cookie'].lower()
        assert c.get('/api/auth/me').json()['username']=='ekinstru'
        assert c.get('/api/models').status_code == 200
        assert c.get('/docs').status_code == 200
        assert c.get('/api/key').status_code == 200
        assert c.post('/api/key/regenerate').status_code == 403
        assert c.post('/api/key/regenerate', headers={'X-CSRF-Token':'wrong'}).status_code == 403
        assert c.post('/api/key/regenerate', headers={'X-CSRF-Token':csrf}).status_code == 200
        assert c.post('/api/auth/logout').status_code == 403
        assert c.post('/api/auth/logout', headers={'X-CSRF-Token':csrf}).status_code == 200
        assert c.get('/api/models').status_code == 401
        assert c.get('/api/auth/me').status_code == 401


def test_external_inference_stays_api_key_only():
    with TestClient(app) as c:
        # Inference endpoints require API key, not dashboard cookie.
        assert c.post('/v1/iris-demo/predict',json={'inputs':[]}).status_code == 401
        r=login(c);csrf=r.json()['csrf_token']
        key=c.get('/api/key').json()['api_key']
        assert c.post('/v1/iris-demo/predict', json={'inputs':{'sepal length (cm)':5.1,'sepal width (cm)':3.5,'petal length (cm)':1.4,'petal width (cm)':0.2}},headers={'X-API-Key':key}).json()['predictions']==[0]
        # Existing model + DB preserved when code updated (tested with same DATA_DIR).
        assert any(m['slug']=='iris-demo' for m in c.get('/api/models').json())
        model=LinearRegression().fit([[1,1],[2,2],[3,3]],[2,4,6])
        buf=io.BytesIO();joblib.dump(model,buf)
        files={'file':('new.joblib',buf.getvalue(),'application/octet-stream')}
        payload={'name':'Updated Model','features':'["a","b"]','trusted':'true'}
        assert c.post('/api/models',files=files,data=payload).status_code == 403
        added=c.post('/api/models',files=files,data=payload,headers={'X-CSRF-Token':csrf})
        assert added.status_code == 201,added.text
        assert c.get('/api/models/updated-model').status_code==200
        assert c.patch('/api/models/updated-model/status',json={'active':False},headers={'X-CSRF-Token':csrf}).status_code==200
        assert c.delete('/api/models/updated-model',headers={'X-CSRF-Token':csrf}).status_code==200
