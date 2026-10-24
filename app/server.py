"""Serveur web local de LoRA Creator. Lancé par « Lancer LoRA Creator.bat » avec le Python de kohya."""

import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import engine as E

app = FastAPI(title="LoRA Creator")
STATIC = Path(__file__).parent / "static"


def guard(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))


# ------------------------------------------------------------------ infos générales

@app.get("/api/info")
def info():
    return {"types": E.TYPES, "data_dir": str(E.DATA), "forge_dir": str(E.FORGE_LORA)}


@app.get("/api/models")
def models():
    return E.list_models()


@app.post("/api/estimate")
def estimate(body: dict = Body(...)):
    return E.estimate(int(body.get("n_images") or 0), body["type"], body["quality"], body.get("overrides"))


@app.post("/api/defaults")
def defaults(body: dict = Body(...)):
    return {f: E.defaults_for(f, body["type"], body["quality"]) for f in ("sd15", "sdxl")}


# ------------------------------------------------------------------ datasets

@app.get("/api/datasets")
def datasets():
    return E.list_datasets()


@app.post("/api/datasets")
def create_dataset(body: dict = Body(...)):
    return {"slug": E.create_dataset(body.get("name") or "dataset")}


@app.get("/api/datasets/{slug}")
def dataset(slug: str):
    return guard(E.dataset_detail, slug)


@app.post("/api/datasets/{slug}/upload")
async def upload(slug: str, files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        results.append(guard(E.add_image, slug, f.filename or "image.png", await f.read()))
    return results


@app.post("/api/datasets/{slug}/import")
def import_folder(slug: str, body: dict = Body(...)):
    return guard(E.import_folder, slug, body["folder"])


@app.put("/api/datasets/{slug}/captions")
def captions(slug: str, body: dict = Body(...)):
    guard(E.save_captions, slug, body)
    return {"ok": True}


@app.post("/api/datasets/{slug}/remove-tag")
def remove_tag(slug: str, body: dict = Body(...)):
    return {"changed": guard(E.remove_tag, slug, body["tag"])}


@app.delete("/api/datasets/{slug}/images/{fname}")
def delete_image(slug: str, fname: str):
    guard(E.delete_image, slug, fname)
    return {"ok": True}


@app.post("/api/datasets/{slug}/autotag")
def autotag(slug: str, body: dict = Body(...)):
    job = guard(E.start_autotag, slug, body.get("method", "wd14"), bool(body.get("overwrite")),
                float(body.get("threshold", 0.35)))
    return job.state()


@app.get("/files/dataset/{slug}/{fname}")
def dataset_file(slug: str, fname: str, thumb: int = 0):
    p = guard(E.dataset_dir, slug) / Path(fname).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(E.thumbnail(p, slug) if thumb else p)


# ------------------------------------------------------------------ entraînement

@app.post("/api/train")
def train(body: dict = Body(...)):
    return guard(E.start_training, body).state()


@app.get("/api/job")
def job():
    j = E.current_job()
    return j.state() if j else None


@app.post("/api/job/cancel")
def cancel():
    j = E.current_job()
    if j:
        j.cancel()
    return {"ok": True}


@app.get("/api/runs")
def runs():
    return E.list_runs()


@app.get("/api/runs/{run_id}")
def run(run_id: str):
    return guard(E.get_run, run_id)


@app.post("/api/runs/{run_id}/forge")
def to_forge(run_id: str, body: dict = Body(...)):
    r = guard(E.get_run, run_id)
    return {"file": guard(E.copy_to_forge, r, body["file"])}


@app.get("/files/run/{run_id}/{kind}/{fname}")
def run_file(run_id: str, kind: str, fname: str, thumb: int = 0):
    base = guard(E._run_dir, run_id) / "output"
    p = (base / "sample" if kind == "sample" else base) / Path(fname).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(E.thumbnail(p, run_id) if thumb else p)


@app.post("/api/open-folder")
def open_folder(body: dict = Body(...)):
    import os
    target = {"data": E.DATA, "lora": E.LORA_OUT, "forge": E.FORGE_LORA}.get(body.get("which"))
    if body.get("run"):
        target = guard(E._run_dir, body["run"]) / "output"
    if target and Path(target).exists():
        os.startfile(target)
    return {"ok": True}


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


if __name__ == "__main__":
    port = int(E.CFG.get("port", 7870))
    url = f"http://127.0.0.1:{port}"
    print(f"\n  LoRA Creator : {url}\n  (ferme cette fenêtre pour arrêter l'appli)\n")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
