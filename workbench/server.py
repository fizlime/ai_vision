"""Local web server. No database or persistent image storage."""
import base64
import binascii
import csv
import io
import json
from pathlib import Path
import threading

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .engine import CATALOG, decode_image, run_pipeline, validate

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Vision Bench")
lock = threading.Lock()
cv2.setNumThreads(2)


@app.middleware("http")
async def local_only(request: Request, call_next):
    if request.headers.get("host", "").split(":")[0] not in ("127.0.0.1", "localhost", "testserver"):
        return JSONResponse({"detail": "Local access only"}, status_code=403)
    origin = request.headers.get("origin")
    if origin and origin not in (f"http://{request.headers.get('host')}", f"https://{request.headers.get('host')}"):
        return JSONResponse({"detail": "Cross-origin access is disabled"}, status_code=403)
    return await call_next(request)


async def body(request):
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > 56 * 1024 * 1024:
            raise HTTPException(413, "요청 크기는 56MB 이하여야 합니다.")
    try:
        parsed = json.loads(data)
        if not isinstance(parsed, dict): raise ValueError()
        return parsed
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "올바른 JSON 객체가 필요합니다.")


def process(payload):
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "다른 이미지가 처리 중입니다. 잠시 후 다시 실행하세요.")
    try:
        encoded = payload.get("image")
        if not isinstance(encoded, str): raise ValueError("이미지를 먼저 선택하세요.")
        image = decode_image(base64.b64decode(encoded, validate=True))
        result, _ = run_pipeline(image, payload.get("recipe"))
        return result
    except (ValueError, TypeError, binascii.Error, cv2.error) as exc:
        raise HTTPException(422, str(exc))
    finally:
        lock.release()


@app.get("/api/catalog")
def catalog():
    return list(CATALOG.values())


@app.get("/api/samples")
def samples():
    return [p.name for p in sorted((ROOT.parent / "test_image").glob("*.tiff"))]


@app.get("/api/samples/{name}")
def sample(name: str):
    if name not in samples(): raise HTTPException(404)
    return FileResponse(ROOT.parent / "test_image" / name)


@app.post("/api/run")
async def run(request: Request):
    return await run_in_threadpool(process, await body(request))


@app.post("/api/validate")
async def validate_recipe(request: Request):
    try: return validate(await body(request))
    except ValueError as exc: raise HTTPException(422, str(exc))


@app.post("/api/export/{format}")
async def export(format: str, request: Request):
    try: recipe = validate(await body(request))
    except ValueError as exc: raise HTTPException(422, str(exc))
    if format == "python":
        source = (ROOT / "engine.py").read_text(encoding="utf-8")
        config = repr(json.dumps(recipe, ensure_ascii=False))
        source += f'''\n\nRECIPE = json.loads({config})\n\nif __name__ == "__main__":\n    import argparse\n    from pathlib import Path\n    parser = argparse.ArgumentParser(description="Vision Bench exported pipeline")\n    parser.add_argument("image", help="Input 8-bit PNG/JPG/TIFF image")\n    parser.add_argument("--output", default="vision_output", help="Output directory")\n    args = parser.parse_args()\n    original = decode_image(Path(args.image).read_bytes())\n    result, processed = run_pipeline(original, RECIPE, include_previews=False)\n    target = Path(args.output)\n    target.mkdir(parents=True, exist_ok=True)\n    ok, encoded = cv2.imencode(".png", processed)\n    if not ok: raise RuntimeError("Could not encode output")\n    (target / "result.png").write_bytes(encoded.tobytes())\n    (target / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")\n    print("Saved:", target.resolve())\n'''
        return Response(source, media_type="text/x-python; charset=utf-8", headers={"Content-Disposition": 'attachment; filename="vision_pipeline.py"'})
    if format == "csv":
        out = io.StringIO(newline="")
        writer = csv.writer(out)
        writer.writerow(["order", "node_id", "operation", "enabled", "parameter", "value"])
        for i, node in enumerate(recipe["nodes"], 1):
            for key, value in (node["params"].items() or [("", "")]):
                row = [i, node["id"], node["type"], node["enabled"], key, value]
                writer.writerow(["'"+str(v) if isinstance(v,str) and v.startswith(("=", "+", "-", "@")) else v for v in row])
        return Response("\ufeff"+out.getvalue(), media_type="text/csv; charset=utf-8")
    raise HTTPException(404)


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="ui")
