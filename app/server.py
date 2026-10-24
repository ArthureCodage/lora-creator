"""Serveur web local de LoRA Creator. Lancé par « Lancer LoRA Creator.bat » (Windows) ou lancer-lora-creator.sh
(Linux) avec le Python de kohya."""

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

# Profil de carte choisi dans le popup : demandé à chaque lancement de l'appli (pas à chaque rechargement de page).
SESSION = {"profile": None, "gpus": None}


@app.get("/api/info")
def info():
    if SESSION["gpus"] is None:
        SESSION["gpus"] = E.detect_gpus()
    return {
        "types": E.TYPES, "data_dir": str(E.DATA), "forge_dir": str(E.FORGE_LORA),
        "profiles": {k: {f: v[f] for f in ("label", "short", "desc")} for k, v in E.PROFILES.items()},
        "profile": SESSION["profile"], "gpus": SESSION["gpus"],
        "recommended": E.recommended_profile(SESSION["gpus"]),
    }


@app.post("/api/profile")
def set_profile(body: dict = Body(...)):
    if body.get("profile") not in E.PROFILES:
        raise HTTPException(400, "Profil inconnu.")
    SESSION["profile"] = body["profile"]
    return {"profile": SESSION["profile"]}


def _profile(body: dict) -> str | None:
    return body.get("profile") or SESSION["profile"]


@app.get("/api/models")
def models():
    return E.list_models()


@app.post("/api/estimate")
def estimate(body: dict = Body(...)):
    return E.estimate(int(body.get("n_images") or 0), body["type"], body["quality"], body.get("overrides"),
                      _profile(body))


@app.post("/api/defaults")
def defaults(body: dict = Body(...)):
    return {f: E.defaults_for(f, body["type"], body["quality"], _profile(body)) for f in ("sd15", "sdxl")}


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
    return guard(E.start_training, {**body, "profile": _profile(body)}).state()


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
    """Ouvre le dossier dans l'explorateur. Sur une machine sans écran (serveur, cloud), renvoie juste le chemin."""
    import os
    import shutil
    import subprocess
    target = {"data": E.DATA, "lora": E.LORA_OUT, "forge": E.FORGE_LORA}.get(body.get("which"))
    if body.get("run"):
        target = guard(E._run_dir, body["run"]) / "output"
    opened = False
    if target and Path(target).exists():
        if E.IS_WIN:
            os.startfile(target)
            opened = True
        elif shutil.which("xdg-open") and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            subprocess.Popen(["xdg-open", str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            opened = True
    return {"ok": True, "opened": opened, "path": str(target) if target else None}


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


def _arg(name: str, default):
    """--host 0.0.0.0 / --port 7870 en ligne de commande (prioritaires sur config.json)."""
    if name in sys.argv and sys.argv.index(name) + 1 < len(sys.argv):
        return sys.argv[sys.argv.index(name) + 1]
    return default


if __name__ == "__main__":
    host = _arg("--host", E.CFG.get("host", "127.0.0.1"))
    port = int(_arg("--port", E.CFG.get("port", 7870)))
    url = f"http://127.0.0.1:{port}"
    print(f"\n  LoRA Creator : {url}" + (f"  (écoute sur {host})" if host != "127.0.0.1" else "") +
          "\n  (ferme cette fenêtre ou Ctrl+C pour arrêter l'appli)\n")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
