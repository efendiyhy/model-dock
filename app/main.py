"""ModelDock — functional local ML-model upload and FastAPI inference demo.

SECURITY: This is a LOCAL single-operator prototype, not safe for public deployment.
Joblib/pickle models execute Python code when deserialized. Upload ONLY trusted files.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import sys
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("MODELDOCK_DATA_DIR", str(ROOT / "data"))).resolve()
MODELS = DATA / "models"
DB_FILE = DATA / "modeldock.sqlite3"
KEY_FILE = DATA / "demo_api_key.txt"
STATIC = ROOT / "app" / "static"
DEMO_MODEL = ROOT / "examples" / "iris_classifier.joblib"
MAX_UPLOAD = 35 * 1024 * 1024
# Folder for Python modules that define custom model classes (e.g. ncs_models.py).
# Pickled models only store object state, so their class's module must be importable.
CUSTOM_MODULES = Path(os.getenv("MODELDOCK_CUSTOM_MODULES_DIR", str(ROOT / "custom_modules"))).resolve()
if CUSTOM_MODULES.is_dir() and str(CUSTOM_MODULES) not in sys.path:
    sys.path.insert(0, str(CUSTOM_MODULES))
LOCK = threading.RLock()
CACHE: dict[str, Any] = {}
# Defaults are specifically for local prototyping. Change these via environment
# variables before any deployment accessible beyond localhost.
ADMIN_USERNAME = os.getenv("MODELDOCK_ADMIN_USERNAME", "ekinstru")
ADMIN_PASSWORD = os.getenv("MODELDOCK_ADMIN_PASSWORD", "sandikuat")
SESSION_COOKIE = "modeldock_admin"
SESSION_LIFETIME = 8 * 60 * 60
LOGIN_ATTEMPTS: dict[str, list[float]] = {}



def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def db():
    with LOCK:
        conn = sqlite3.connect(DB_FILE, timeout=20)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:65] or "model"


def as_json(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [as_json(x) for x in v]
    if isinstance(v, dict):
        return {str(k): as_json(val) for k, val in v.items()}
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def read_model_file(path: Path | str) -> Any:
    """joblib.load with a clear message when a custom class's module is missing."""
    try:
        return joblib.load(path)  # Trusted files ONLY.
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise ValueError(
            f"Model memakai class kustom dari modul '{missing}' yang tidak ada di server. "
            f"Salin file '{missing.split('.')[0]}.py' (kode class yang dipakai saat training) "
            f"ke folder custom_modules/ lalu restart server."
        ) from exc
    except AttributeError as exc:
        if "Can't get attribute" in str(exc) or "can't get attribute" in str(exc).lower():
            raise ValueError(
                f"Class kustom tidak ditemukan ({exc}). Jika class didefinisikan di notebook (__main__), "
                "pindahkan ke file .py, simpan di folder custom_modules/, import dari file tersebut, "
                "lalu latih/simpan ulang model."
            ) from exc
        raise


def inspect_estimator(model: Any, manual_features: str = "") -> dict:
    if not callable(getattr(model, "predict", None)):
        raise ValueError("File bukan model yang menyediakan method predict().")
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        feature_names = getattr(model, "feature_names", None)
    count = getattr(model, "n_features_in_", None)
    if manual_features.strip():
        try:
            features = json.loads(manual_features)
        except json.JSONDecodeError:
            raise ValueError('Format fitur harus JSON, contoh: ["age", "income"].')
        if not isinstance(features, list) or not features or any(not isinstance(f, str) or not f.strip() for f in features):
            raise ValueError("Fitur harus berupa JSON array berisi nama-nama fitur.")
    elif feature_names is not None and len(feature_names):
        features = [str(f) for f in feature_names]
    elif count is not None and int(count) > 0:
        features = [f"feature_{i+1}" for i in range(int(count))]
    else:
        raise ValueError('Nama fitur tidak ditemukan. Isi kolom fitur manual: ["feature_1", "feature_2"].')
    if len(set(features)) != len(features):
        raise ValueError("Nama fitur tidak boleh duplikat.")
    if count is not None and len(features) != int(count):
        raise ValueError(f"Model membutuhkan {int(count)} fitur, tetapi diberikan {len(features)}.")
    if len(features) > 1000:
        raise ValueError("Maksimum 1000 fitur untuk prototipe ini.")
    classes = getattr(model, "classes_", None)
    kind = "classification" if classes is not None else "regression"
    sample = {name: 0 for name in features}
    return {"features": features, "task": kind, "classes": as_json(classes) if classes is not None else [], "sample_input": sample}


def serialize_row(row: sqlite3.Row) -> dict:
    result = dict(row)
    for field in ("features", "classes", "sample_input"):
        result[field] = json.loads(result[field])
    result["active"] = bool(result["active"])
    result.pop("file_path", None)
    return result


def get_model(slug: str) -> sqlite3.Row:
    with db() as conn:
        row = conn.execute("SELECT * FROM models WHERE slug=?", (slug,)).fetchone()
    if row is None:
        raise HTTPException(404, "Model tidak ditemukan")
    return row


def load_model(row: sqlite3.Row):
    key = row["slug"]
    with LOCK:
        if key not in CACHE:
            CACHE[key] = read_model_file(row["file_path"])  # Trusted files ONLY.
        return CACHE[key]


def new_key() -> str:
    return "md_live_" + secrets.token_urlsafe(25)


def initialize() -> None:
    first_start = not DB_FILE.exists()
    MODELS.mkdir(parents=True, exist_ok=True)
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS models (
            slug TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
            filename TEXT NOT NULL, file_path TEXT NOT NULL, task TEXT NOT NULL,
            features TEXT NOT NULL, classes TEXT NOT NULL, sample_input TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL,
            framework TEXT NOT NULL DEFAULT 'scikit-learn / joblib', size_bytes INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT, model_slug TEXT NOT NULL,
            at TEXT NOT NULL, status INTEGER NOT NULL, duration_ms REAL NOT NULL,
            source TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_requests_at ON requests(at);
        CREATE TABLE IF NOT EXISTS admin_sessions (
            session_hash TEXT PRIMARY KEY,
            csrf_token TEXT NOT NULL,
            expires_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_admin_session_expiry ON admin_sessions(expires_at);
        """)
    if not KEY_FILE.exists():
        KEY_FILE.write_text(new_key(), encoding="utf8")
        try:
            KEY_FILE.chmod(0o600)
        except OSError:
            pass
    # A real pre-trained sample model, bundled for immediately testing predictions.
    with db() as conn:
        exists = conn.execute("SELECT 1 FROM models LIMIT 1").fetchone()
        if first_start and not exists and DEMO_MODEL.exists():
            target = MODELS / "iris-demo.joblib"
            shutil.copyfile(DEMO_MODEL, target)
            model = joblib.load(target)
            meta = inspect_estimator(model)
            meta["sample_input"] = {
                "sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
                "petal length (cm)": 1.4, "petal width (cm)": 0.2,
            }
            conn.execute("""INSERT INTO models(slug,name,description,filename,file_path,task,features,classes,sample_input,active,created_at,framework,size_bytes)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    "iris-demo", "Iris Species Classifier",
                    "Pre-trained sample · Iris flower classification (3 species)", "iris_classifier.joblib", str(target), meta["task"],
                    json.dumps(meta["features"]), json.dumps(meta["classes"]),
                    json.dumps(meta["sample_input"]), 1, utcnow(), "Scikit-learn", target.stat().st_size,
                ))


initialize()
app = FastAPI(title="ModelDock · ML Inference API", version="0.1.0", description="Local prototype: upload a trusted trained ML model and expose an authenticated prediction endpoint.")
origins = [x.strip() for x in os.getenv("MODELDOCK_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["X-API-Key", "Content-Type"])
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def session_from_request(request: Request) -> dict | None:
    raw = request.cookies.get(SESSION_COOKIE, "")
    if not raw or len(raw) > 180:
        return None
    hashed = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    with db() as conn:
        entry = conn.execute(
            "SELECT csrf_token, expires_at FROM admin_sessions WHERE session_hash=?",
            (hashed,),
        ).fetchone()
    if entry is None or entry["expires_at"] <= int(time.time()):
        return None
    return {"csrf": entry["csrf_token"], "session_hash": hashed}


@app.middleware("http")
async def protect_admin_routes(request: Request, call_next):
    path = request.url.path
    # Public inference endpoints intentionally continue to authenticate via X-API-Key.
    # Admin functions must be protected on the SERVER, not just hidden by the UI.
    admin_route = path.startswith("/api/") and not path.startswith("/api/auth/")
    admin_docs = path in ("/docs", "/redoc", "/openapi.json")
    if admin_route or admin_docs:
        session = session_from_request(request)
        if session is None:
            return JSONResponse({"detail": "Silakan login sebagai admin."}, status_code=401)
        if admin_route and request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            supplied = request.headers.get("x-csrf-token", "")
            if not supplied or not secrets.compare_digest(supplied, session["csrf"]):
                return JSONResponse({"detail": "Token keamanan tidak valid. Refresh dan login ulang."}, status_code=403)
    return await call_next(request)


class AdminLogin(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login", include_in_schema=False)
def admin_login(body: AdminLogin, request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    with LOCK:
        recent = [t for t in LOGIN_ATTEMPTS.get(ip, []) if now - t < 600]
        LOGIN_ATTEMPTS[ip] = recent
        if len(recent) >= 6:
            raise HTTPException(429, "Terlalu banyak percobaan login. Coba lagi dalam 10 menit.")
    # Constant-time comparisons. Credentials remain server-side only.
    username_ok = secrets.compare_digest(body.username.encode(), ADMIN_USERNAME.encode())
    password_ok = secrets.compare_digest(body.password.encode(), ADMIN_PASSWORD.encode())
    if not (username_ok and password_ok):
        with LOCK:
            LOGIN_ATTEMPTS[ip].append(now)
        raise HTTPException(401, "Username atau password salah.")
    with LOCK:
        LOGIN_ATTEMPTS.pop(ip, None)
    session_id = secrets.token_urlsafe(36)
    csrf_token = secrets.token_urlsafe(32)
    session_hash = hashlib.sha256(session_id.encode()).hexdigest()
    expiry = int(now) + SESSION_LIFETIME
    with db() as conn:
        conn.execute("DELETE FROM admin_sessions WHERE expires_at<=?", (int(now),))
        conn.execute("INSERT INTO admin_sessions VALUES (?,?,?)", (session_hash, csrf_token, expiry))
    response = JSONResponse({"ok": True, "username": ADMIN_USERNAME, "csrf_token": csrf_token})
    secure = os.getenv("MODELDOCK_COOKIE_SECURE", "").lower() in ("1", "true", "yes") or request.url.scheme == "https"
    response.set_cookie(SESSION_COOKIE, session_id, max_age=SESSION_LIFETIME,
                        httponly=True, secure=secure, samesite="strict", path="/")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/auth/me", include_in_schema=False)
def admin_me(request: Request):
    session = session_from_request(request)
    if session is None:
        raise HTTPException(401, "Belum login.")
    response = JSONResponse({"username": ADMIN_USERNAME, "csrf_token": session["csrf"]})
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/api/auth/logout", include_in_schema=False)
def admin_logout(request: Request):
    session = session_from_request(request)
    if session is None:
        raise HTTPException(401, "Belum login.")
    supplied = request.headers.get("x-csrf-token", "")
    if not supplied or not secrets.compare_digest(supplied, session["csrf"]):
        raise HTTPException(403, "Token keamanan tidak valid.")
    with db() as conn:
        conn.execute("DELETE FROM admin_sessions WHERE session_hash=?", (session["session_hash"],))
    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response



@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "ModelDock", "timestamp": utcnow()}


@app.get("/api/overview", include_in_schema=False)
def overview():
    with db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
        running = conn.execute("SELECT COUNT(*) FROM models WHERE active=1").fetchone()[0]
        count = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        avg = conn.execute("SELECT AVG(duration_ms) FROM requests").fetchone()[0]
        latest = [dict(r) for r in conn.execute("SELECT * FROM requests ORDER BY id DESC LIMIT 7")]
        days = [dict(r) for r in conn.execute("SELECT substr(at,1,10) AS day, COUNT(*) AS requests FROM requests GROUP BY day ORDER BY day DESC LIMIT 7")]
    return {"models": total, "running": running, "requests": count, "latency": round(avg or 0, 1), "latest": latest, "daily": list(reversed(days)), "uptime": "Online"}


@app.get("/api/models", include_in_schema=False)
def models_list():
    with db() as conn:
        rows = conn.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
        counts = {r["model_slug"]: r["n"] for r in conn.execute("SELECT model_slug, COUNT(*) AS n FROM requests GROUP BY model_slug")}
    return [{**serialize_row(r), "requests": counts.get(r["slug"], 0)} for r in rows]


@app.get("/api/models/{slug}", include_in_schema=False)
def model_detail(slug: str):
    return serialize_row(get_model(slug))


@app.post("/api/models", status_code=201, include_in_schema=False)
async def add_model(file: UploadFile = File(...), name: str = Form(...), description: str = Form(""), features: str = Form(""), trusted: bool = Form(False)):
    if not trusted:
        raise HTTPException(400, "Konfirmasi bahwa model adalah file tepercaya sebelum upload.")
    if not (1 <= len(name.strip()) <= 80):
        raise HTTPException(400, "Nama model harus 1–80 karakter.")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in (".joblib", ".pkl", ".pickle"):
        raise HTTPException(400, "Saat ini hanya file .joblib, .pkl, dan .pickle yang didukung.")
    base = slugify(name)
    with db() as conn:
        if conn.execute("SELECT 1 FROM models WHERE slug=?", (base,)).fetchone():
            raise HTTPException(409, "Nama/slug model sudah digunakan. Pilih nama lain.")
    target = MODELS / f"{uuid.uuid4().hex}{ext}"
    length = 0
    try:
        with target.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                length += len(chunk)
                if length > MAX_UPLOAD:
                    raise HTTPException(413, "Ukuran model maksimal 35 MB untuk prototipe.")
                out.write(chunk)
        if not length:
            raise HTTPException(400, "File model kosong.")
        # Deserialization is potentially arbitrary code execution: TRUSTED LOCAL FILES ONLY.
        try:
            estimator = read_model_file(target)
            meta = inspect_estimator(estimator, features)
        except Exception as exc:
            raise HTTPException(422, f"Model tidak bisa dibaca: {str(exc)[:280]}") from exc
        with db() as conn:
            conn.execute("""INSERT INTO models(slug,name,description,filename,file_path,task,features,classes,sample_input,active,created_at,framework,size_bytes)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                base, name.strip(), description[:500], Path(file.filename or "model").name,
                str(target), meta["task"], json.dumps(meta["features"]),
                json.dumps(meta["classes"]), json.dumps(meta["sample_input"]), 1,
                utcnow(), "Python · joblib", length,
            ))
        with LOCK:
            CACHE[base] = estimator
        return {"message": "Model berhasil di-deploy", "slug": base, "endpoint": f"/v1/{base}/predict", "model": serialize_row(get_model(base))}
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


class StatusBody(BaseModel):
    active: bool


@app.patch("/api/models/{slug}/status", include_in_schema=False)
def update_status(slug: str, body: StatusBody):
    get_model(slug)
    with db() as conn:
        conn.execute("UPDATE models SET active=? WHERE slug=?", (int(body.active), slug))
    if not body.active:
        with LOCK:
            CACHE.pop(slug, None)
    return {"active": body.active, "slug": slug}


@app.delete("/api/models/{slug}", include_in_schema=False)
def delete_model(slug: str):
    row = get_model(slug)
    with db() as conn:
        conn.execute("DELETE FROM models WHERE slug=?", (slug,))
    with LOCK:
        CACHE.pop(slug, None)
    Path(row["file_path"]).unlink(missing_ok=True)
    return {"deleted": slug}


@app.get("/api/requests", include_in_schema=False)
def request_history(limit: int = 80):
    limit = max(1, min(limit, 500))
    with db() as conn:
        rows = conn.execute("SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/key", include_in_schema=False)
def show_key():
    return {"api_key": KEY_FILE.read_text(encoding="utf8").strip(), "header": "X-API-Key"}


@app.post("/api/key/regenerate", include_in_schema=False)
def regenerate_key():
    key = new_key()
    KEY_FILE.write_text(key, encoding="utf8")
    return {"api_key": key, "message": "Kunci lama tidak berlaku lagi"}


class PredictBody(BaseModel):
    inputs: Any = Field(..., description="Object with feature keys, list of objects, or numeric array in model feature order")


def normalize_inputs(raw: Any, features: list[str]) -> pd.DataFrame:
    if isinstance(raw, dict):
        records = [raw]
    elif isinstance(raw, list) and raw:
        if isinstance(raw[0], dict):
            records = raw
        elif isinstance(raw[0], list):
            records = [dict(zip(features, r)) for r in raw if isinstance(r, list) and len(r) == len(features)]
            if len(records) != len(raw):
                raise HTTPException(422, f"Setiap baris harus memiliki tepat {len(features)} nilai.")
        elif len(raw) == len(features):
            records = [dict(zip(features, raw))]
        else:
            raise HTTPException(422, "Format inputs tidak sesuai dengan jumlah fitur.")
    else:
        raise HTTPException(422, "inputs harus object, array nilai, atau array object.")
    if len(records) > 100:
        raise HTTPException(422, "Maksimal 100 baris per request.")
    for i, rec in enumerate(records):
        missing = [f for f in features if f not in rec]
        if missing:
            raise HTTPException(422, f"Baris {i+1}: fitur hilang: {', '.join(missing[:10])}")
        if any(isinstance(rec.get(f), (list, dict)) for f in features):
            raise HTTPException(422, "Tipe fitur harus string, angka, bool, atau null.")
    return pd.DataFrame([{k: rec[k] for k in features} for rec in records], columns=features)


def predict_for(slug: str, inputs: Any, source: str) -> dict:
    started = time.perf_counter()
    status = 200
    try:
        row = get_model(slug)
        if not row["active"]:
            raise HTTPException(503, "Model sedang dihentikan. Aktifkan melalui dashboard.")
        features = json.loads(row["features"])
        estimator = load_model(row)
        frame = normalize_inputs(inputs, features)
        # Support pipelines that use DataFrame column names and ndarray-style models.
        try:
            preds = estimator.predict(frame)
        except (TypeError, ValueError) as exc:
            # For positional-only models that were trained without DataFrame names.
            if "feature names" in str(exc).lower():
                preds = estimator.predict(frame.to_numpy())
            else:
                raise
        output = {"model": slug, "task": row["task"], "predictions": as_json(np.asarray(preds).tolist()), "rows": len(frame)}
        if callable(getattr(estimator, "predict_proba", None)):
            try:
                probabilities = estimator.predict_proba(frame)
                output["probabilities"] = as_json(np.asarray(probabilities).round(6).tolist())
                if row["classes"]:
                    output["classes"] = json.loads(row["classes"])
            except (TypeError, ValueError):
                pass
        output["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return output
    except HTTPException as exc:
        status = exc.status_code
        raise
    except Exception as exc:
        status = 422
        raise HTTPException(422, f"Prediksi gagal: {str(exc)[:280]}") from exc
    finally:
        duration = round((time.perf_counter() - started) * 1000, 2)
        with db() as conn:
            conn.execute("INSERT INTO requests(model_slug,at,status,duration_ms,source) VALUES(?,?,?,?,?)", (slug, utcnow(), status, duration, source))


@app.post("/api/models/{slug}/test", include_in_schema=False)
def test_model(slug: str, body: PredictBody):
    return predict_for(slug, body.inputs, "playground")


@app.post("/v1/{slug}/predict", summary="Run inference", tags=["Machine Learning Inference"])
def public_prediction(slug: str, body: PredictBody, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    expected = KEY_FILE.read_text(encoding="utf8").strip()
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(401, "API key tidak valid. Sertakan header X-API-Key.")
    return predict_for(slug, body.inputs, "api")


@app.get("/v1/{slug}/schema", summary="Get feature schema", tags=["Machine Learning Inference"])
def public_schema(slug: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    expected = KEY_FILE.read_text(encoding="utf8").strip()
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(401, "API key tidak valid.")
    row = get_model(slug)
    return {"slug": slug, "task": row["task"], "features": json.loads(row["features"]), "sample_input": json.loads(row["sample_input"]), "active": bool(row["active"])}
