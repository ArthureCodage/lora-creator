"""Moteur de LoRA Creator : presets, estimation du temps, datasets, lancement de kohya (sd-scripts)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import statistics
import struct
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

import toml
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
IS_WIN = os.name == "nt"
_NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if IS_WIN else {}  # pas de console qui clignote


def _load_config() -> dict:
    """config.json (créé depuis config.example.json au 1er lancement). LORA_CREATOR_CONFIG permet d'en pointer un autre."""
    path = Path(os.environ.get("LORA_CREATOR_CONFIG") or APP_DIR.parent / "config.json")
    if not path.exists():
        shutil.copy(APP_DIR.parent / "config.example.json", path)
        print(f"  config.json créé depuis config.example.json — vérifie les chemins : {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    # « ~ » et %USERPROFILE% marchent sous Windows comme sous Linux
    expand = lambda s: os.path.expanduser(os.path.expandvars(s.replace("%USERPROFILE%", "~")))
    for k in ("kohya_dir", "data_dir", "forge_lora_dir"):
        cfg[k] = expand(cfg[k])
    if cfg.get("python"):
        cfg["python"] = expand(cfg["python"])
    cfg["model_dirs"] = [expand(d) for d in cfg.get("model_dirs", [])]
    return cfg


CFG = _load_config()

KOHYA = Path(CFG["kohya_dir"])
SD_SCRIPTS = KOHYA / "sd-scripts"
PYTHON = Path(CFG.get("python") or (KOHYA / "venv" / ("Scripts/python.exe" if IS_WIN else "bin/python")))
DATA = Path(CFG["data_dir"])
DATASETS = DATA / "datasets"
RUNS = DATA / "runs"
LORA_OUT = DATA / "LoRA"
CACHE = DATA / "cache"
FORGE_LORA = Path(CFG["forge_lora_dir"])

for d in (DATASETS, RUNS, LORA_OUT, CACHE / "thumbs", DATA / "models"):
    d.mkdir(parents=True, exist_ok=True)

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# --------------------------------------------------------------------------- presets

TYPES = {
    "personnage": {"label": "Personnage / personne", "class": "person"},
    "style": {"label": "Style artistique", "class": "style"},
    "objet": {"label": "Objet / concept", "class": "object"},
}

QUALITY_EPOCHS = {"rapide": 6, "equilibre": 10, "qualite": 14}

# Nombre de pas visé selon la base, le type et la qualité.
TARGET_STEPS = {
    "sd15": {
        "personnage": {"rapide": 700, "equilibre": 1400, "qualite": 2400},
        "style": {"rapide": 1000, "equilibre": 1800, "qualite": 3000},
        "objet": {"rapide": 600, "equilibre": 1200, "qualite": 2000},
    },
    "sdxl": {
        "personnage": {"rapide": 500, "equilibre": 1000, "qualite": 1600},
        "style": {"rapide": 700, "equilibre": 1300, "qualite": 2000},
        "objet": {"rapide": 450, "equilibre": 900, "qualite": 1400},
    },
}

DIM_ALPHA = {
    "sd15": {"personnage": (32, 16), "style": (64, 32), "objet": (32, 16)},
    "sdxl": {"personnage": (32, 16), "style": (32, 16), "objet": (32, 16)},
}

BASE_RES = {"sd15": 512, "sdxl": 1024}

# Profils de carte graphique, choisis au démarrage de l'appli.
# Les temps sont des valeurs de départ, remplacées par les mesures des vrais entraînements (par profil).
PROFILES = {
    "small": {
        "label": "Ma RTX 2060 (6 Go)",
        "short": "RTX 2060",
        "desc": "Économie de mémoire maximale : fp16, gradient checkpointing, SDXL en fp8, une image à la fois.",
        "batch": {"sd15": 1, "sdxl": 1},
        "unet_lr": {"sd15": 1e-4, "sdxl": 1e-4},
        "te_lr": 5e-5,
        "samples": {"sd15": True, "sdxl": False},  # en SDXL une image de test prend ~9 min sur 6 Go
        "resolutions": {"sd15": [512, 640, 768], "sdxl": [768, 896, 1024]},
        "spi": {"sd15": 0.6, "sdxl": 4.5},  # secondes par pas à la résolution de base
        "overhead": {"sd15": 90, "sdxl": 300},  # chargement du modèle + mise en cache
        "sample_sec": {"sd15": 5, "sdxl": 540},  # par image de test
    },
    "big": {
        "label": "Gros GPU (24 Go et plus, ex. RTX 4090)",
        "short": "Gros GPU",
        "desc": "Vitesse maximale : bf16, sans économies de mémoire, plusieurs images à la fois, images de test partout.",
        "batch": {"sd15": 4, "sdxl": 2},
        "unet_lr": {"sd15": 2e-4, "sdxl": 1.5e-4},  # LR monté avec la taille de batch
        "te_lr": 1e-4,
        "samples": {"sd15": True, "sdxl": True},
        "resolutions": {"sd15": [512, 640, 768], "sdxl": [768, 896, 1024, 1280]},
        "spi": {"sd15": 0.35, "sdxl": 1.0},
        "overhead": {"sd15": 60, "sdxl": 120},
        "sample_sec": {"sd15": 2, "sdxl": 8},
    },
}
DEFAULT_PROFILE = "small"


def prof(profile: str | None) -> dict:
    return PROFILES.get(profile or DEFAULT_PROFILE, PROFILES[DEFAULT_PROFILE])


def detect_gpus() -> list[dict]:
    """Cartes NVIDIA vues par nvidia-smi (sans charger torch, qui garderait de la VRAM)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, **_NO_WINDOW).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        name, _, mem = line.rpartition(",")
        try:
            gpus.append({"name": name.strip(), "vram_gb": round(int(mem) / 1024)})
        except ValueError:
            pass
    return gpus


def recommended_profile(gpus: list[dict]) -> str:
    return "big" if gpus and max(g["vram_gb"] for g in gpus) >= 16 else "small"


def defaults_for(family: str, ltype: str, quality: str, profile: str | None = None) -> dict:
    pr = prof(profile)
    dim, alpha = DIM_ALPHA[family][ltype]
    return {
        "epochs": QUALITY_EPOCHS[quality],
        "target_steps": TARGET_STEPS[family][ltype][quality],
        "dim": dim,
        "alpha": alpha,
        "unet_lr": pr["unet_lr"][family],
        "te_lr": pr["te_lr"] if family == "sd15" else 0.0,
        "resolution": BASE_RES[family],
        "resolutions": pr["resolutions"][family],
        "batch": pr["batch"][family],
        "seed": 1234,
        "samples": pr["samples"][family],
    }


def plan_steps(n_images: int, target: int, epochs: int, repeats: int | None = None, batch: int = 1) -> dict:
    """target = nombre d'images vues visé (équivalent pas à batch 1) ; les pas réels sont divisés par le batch."""
    n = max(1, n_images)
    if not repeats:
        repeats = max(1, round(target / (n * epochs)))
        if repeats == 1 and n * epochs > target * 1.3:  # gros dataset : moins d'epochs
            epochs = max(3, round(target / n))
        repeats = min(repeats, 50)
    save_every = max(1, round(epochs / 5))
    steps = -(-n * repeats // max(1, batch)) * epochs
    return {"repeats": repeats, "epochs": epochs, "steps": steps, "batch": batch, "save_every": save_every}


# --------------------------------------------------------------------------- modèles

_model_cache: dict[str, tuple[float, str | None]] = {}


def detect_family(path: Path) -> str | None:
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 200_000_000:
                return None
            keys = json.loads(f.read(n)).keys()
    except Exception:
        return None
    if any(k.startswith("conditioner.embedders.1") for k in keys):
        return "sdxl"
    if any(k.startswith("model.diffusion_model.") for k in keys):
        if any(k.startswith("cond_stage_model.model.") for k in keys):
            return None  # SD 2.x, non géré
        return "sd15"
    return None


def list_models() -> list[dict]:
    out = []
    for d in CFG["model_dirs"]:
        d = Path(d)
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.safetensors")):
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_size < 1_500_000_000:
                continue
            key = str(p)
            if key not in _model_cache or _model_cache[key][0] != st.st_mtime:
                _model_cache[key] = (st.st_mtime, detect_family(p))
            fam = _model_cache[key][1]
            if fam:
                out.append({
                    "path": key,
                    "name": p.stem,
                    "family": fam,
                    "pony": "pony" in p.stem.lower(),
                    "size_gb": round(st.st_size / 1e9, 1),
                    "folder": d.name,
                })
    out.sort(key=lambda m: (m["family"], m["name"].lower()))
    return out


# --------------------------------------------------------------------------- estimation du temps

def _calibration(profile: str | None) -> dict:
    """s/pas et frais fixes mesurés sur les entraînements terminés de ce profil (médiane des 5 derniers)."""
    profile = profile or DEFAULT_PROFILE
    pr = prof(profile)
    spi: dict[str, list] = {"sd15": [], "sdxl": []}
    ovh: dict[str, list] = {"sd15": [], "sdxl": []}
    for r in list_runs():
        m = r.get("measured")
        if r.get("status") != "finished" or not m or not m.get("reliable") or r.get("family") not in spi:
            continue
        if r.get("profile", "small") != profile or r["params"].get("batch", 1) != pr["batch"][r["family"]]:
            continue
        fam = r["family"]
        res = r["params"]["resolution"]
        spi[fam].append(m["spi"] * (BASE_RES[fam] / res) ** 2)
        ovh[fam].append(m["overhead"])
    cal = {}
    for fam in spi:
        cal[fam] = {
            "spi": statistics.median(spi[fam][:5]) if spi[fam] else pr["spi"][fam],
            "overhead": statistics.median(ovh[fam][:5]) if ovh[fam] else pr["overhead"][fam],
            "calibrated": bool(spi[fam]),
            "runs": len(spi[fam]),
        }
    return cal


def _params(fam: str, ltype: str, quality: str, profile: str | None, adv: dict) -> tuple[dict, dict]:
    p = defaults_for(fam, ltype, quality, profile)
    ov = {k: v for k, v in (adv or {}).items() if v not in (None, "")}
    p.update(ov)
    return p, ov


def estimate(n_images: int, ltype: str, quality: str, overrides: dict | None = None, profile: str | None = None) -> dict:
    cal = _calibration(profile)
    pr = prof(profile)
    result = {}
    for fam in ("sd15", "sdxl"):
        p, ov = _params(fam, ltype, quality, profile, (overrides or {}).get(fam, {}))
        plan = plan_steps(n_images, int(p["target_steps"]), int(p["epochs"]), int(ov.get("repeats") or 0) or None,
                          int(p["batch"]))
        res = int(p["resolution"])
        spi = cal[fam]["spi"] * (res / BASE_RES[fam]) ** 2
        n_prompts = 2 if fam == "sd15" else 1
        sample_events = plan["epochs"] // plan["save_every"] if p["samples"] else 0
        seconds = cal[fam]["overhead"] + plan["steps"] * spi + sample_events * n_prompts * pr["sample_sec"][fam]
        result[fam] = {**plan, "seconds": round(seconds), "calibrated": cal[fam]["calibrated"], "params": p}
    return result


# --------------------------------------------------------------------------- datasets

def slugify(name: str) -> str:
    s = re.sub(r"[^\w\- ]", "", name, flags=re.UNICODE).strip().replace(" ", "_")
    return s[:60] or "lora"


def dataset_dir(slug: str) -> Path:
    p = (DATASETS / slug).resolve()
    if p.parent != DATASETS.resolve():
        raise ValueError("nom de dataset invalide")
    return p


def list_images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.is_dir() else []


def list_datasets() -> list[dict]:
    out = []
    for d in sorted(DATASETS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        imgs = list_images(d)
        meta = _read_json(d / "dataset.json")
        out.append({
            "slug": d.name,
            "name": meta.get("name", d.name),
            "n_images": len(imgs),
            "n_captioned": sum(1 for i in imgs if i.with_suffix(".txt").exists()),
            "thumb": imgs[0].name if imgs else None,
        })
    return out


def create_dataset(name: str) -> str:
    slug = slugify(name)
    base, i = slug, 2
    while (DATASETS / slug).exists():
        slug, i = f"{base}_{i}", i + 1
    d = DATASETS / slug
    d.mkdir(parents=True)
    _write_json(d / "dataset.json", {"name": name, "created": _now()})
    return slug


def dataset_detail(slug: str) -> dict:
    d = dataset_dir(slug)
    imgs = []
    for p in list_images(d):
        cap = p.with_suffix(".txt")
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            w = h = 0
        imgs.append({
            "file": p.name,
            "caption": cap.read_text(encoding="utf-8").strip() if cap.exists() else "",
            "w": w, "h": h,
            "small": min(w, h) < 512,
        })
    meta = _read_json(d / "dataset.json")
    return {"slug": slug, "name": meta.get("name", slug), "path": str(d), "images": imgs}


MAX_SIDE = 2048


def add_image(slug: str, filename: str, data: bytes) -> dict:
    """Prépare une image : orientation, RGB, taille max, doublons ignorés."""
    d = dataset_dir(slug)
    digest = hashlib.md5(data).hexdigest()
    hashes_file = d / ".hashes.json"
    hashes = _read_json(hashes_file)
    if digest in hashes:
        return {"file": hashes[digest], "status": "doublon"}
    from io import BytesIO
    try:
        im = Image.open(BytesIO(data))
        im = ImageOps.exif_transpose(im)
    except Exception:
        return {"file": filename, "status": "illisible"}
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    stem = re.sub(r"[^\w\-]", "_", Path(filename).stem)[:50] or "img"
    is_jpg = Path(filename).suffix.lower() in (".jpg", ".jpeg")
    ext = ".jpg" if is_jpg else ".png"
    target, i = d / f"{stem}{ext}", 2
    while target.exists():
        target, i = d / f"{stem}_{i}{ext}", i + 1
    if is_jpg:
        im.save(target, quality=95)
    else:
        im.save(target)
    hashes[digest] = target.name
    _write_json(hashes_file, hashes)
    return {"file": target.name, "status": "ok", "small": min(im.size) < 512}


def import_folder(slug: str, folder: str) -> dict:
    src = Path(folder.strip().strip('"'))
    if not src.is_dir():
        raise ValueError(f"Dossier introuvable : {src}")
    added, dup = 0, 0
    for p in list_images(src):
        r = add_image(slug, p.name, p.read_bytes())
        if r["status"] == "ok":
            added += 1
            cap = p.with_suffix(".txt")
            if cap.exists():
                (dataset_dir(slug) / r["file"]).with_suffix(".txt").write_text(
                    cap.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        elif r["status"] == "doublon":
            dup += 1
    return {"added": added, "duplicates": dup}


def save_captions(slug: str, captions: dict[str, str]) -> None:
    d = dataset_dir(slug)
    for fname, text in captions.items():
        p = d / Path(fname).name
        if p.exists():
            p.with_suffix(".txt").write_text(text.strip(), encoding="utf-8")


def remove_tag(slug: str, tag: str) -> int:
    tag = tag.strip().lower()
    n = 0
    for p in list_images(dataset_dir(slug)):
        cap = p.with_suffix(".txt")
        if not cap.exists():
            continue
        tags = [t.strip() for t in cap.read_text(encoding="utf-8").split(",")]
        kept = [t for t in tags if t and t.lower() != tag]
        if len(kept) != len([t for t in tags if t]):
            cap.write_text(", ".join(kept), encoding="utf-8")
            n += 1
    return n


def delete_image(slug: str, fname: str) -> None:
    p = dataset_dir(slug) / Path(fname).name
    for f in (p, p.with_suffix(".txt"), p.with_suffix(".npz")):
        if f.exists():
            f.unlink()
    hashes_file = dataset_dir(slug) / ".hashes.json"
    hashes = {k: v for k, v in _read_json(hashes_file).items() if v != p.name}
    _write_json(hashes_file, hashes)


def thumbnail(path: Path, key: str) -> Path:
    t = CACHE / "thumbs" / (hashlib.md5(f"{path}{path.stat().st_mtime}".encode()).hexdigest() + ".jpg")
    if not t.exists():
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((320, 320))
            im.save(t, quality=85)
    return t


# --------------------------------------------------------------------------- jobs (un seul à la fois : le GPU)

STEP_RE = re.compile(
    r"(?:steps:\s*)?(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:?]+),\s*([\d.]+)\s*(s/it|it/s)(?:.*?avr_loss=([\d.]+))?")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
EPOCH_RE = re.compile(r"\bepoch (\d+)/(\d+)")
PHASES = [
    ("caching latents", "Mise en cache des images"),
    ("caching text encoder", "Mise en cache des légendes"),
    ("loading", "Chargement du modèle"),
    ("load ", "Chargement du modèle"),
    ("prepare optimizer", "Préparation"),
    ("generating sample", "Génération des images de test"),
    ("saving checkpoint", "Sauvegarde"),
    ("downloading", "Téléchargement du modèle de tagging"),
]


class Job:
    def __init__(self, kind: str, cmd: list[str], log_path: Path, run_id: str | None = None, on_end=None):
        self.id = uuid.uuid4().hex[:8]
        self.kind = kind
        self.run_id = run_id
        self.cmd = cmd
        self.log_path = log_path
        self.on_end = on_end
        self.lines: deque[str] = deque(maxlen=300)
        self.phase = "Démarrage"
        self.status = "running"
        self.error = None
        self.step = self.total = 0
        self.epoch = self.epochs = 0
        self.rate = None  # secondes par pas (moyenne depuis le début)
        self.eta = None
        self.loss = None
        self.started = time.time()
        self.first_step_at = None
        self.ended = None
        self.proc: subprocess.Popen | None = None
        self.cancelled = False
        self.final_status = None

    def start(self):
        env = os.environ.copy()
        env.update({"COLUMNS": "160", "TF_CPP_MIN_LOG_LEVEL": "2", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
                    "PATH": str(PYTHON.parent) + os.pathsep + env.get("PATH", "")})
        # processus dans son propre groupe, pour pouvoir l'arrêter avec tous ses enfants
        group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if IS_WIN
                 else {"start_new_session": True})
        self.proc = subprocess.Popen(
            self.cmd, cwd=str(SD_SCRIPTS), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, **group)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        buf = ""
        with open(self.log_path, "a", encoding="utf-8") as log:
            log.write("> " + " ".join(self.cmd) + "\n")
            while True:
                chunk = os.read(self.proc.stdout.fileno(), 8192)
                if not chunk:
                    break
                buf += ANSI_RE.sub("", chunk.decode("utf-8", errors="replace").replace("\x00", ""))
                parts = re.split(r"[\r\n]", buf)
                buf = parts.pop()
                for line in parts:
                    if line.strip():
                        self._line(line, log)
            if buf.strip():
                self._line(buf, log)
        code = self.proc.wait()
        self.ended = time.time()
        if self.cancelled:
            self.final_status = "cancelled"
        elif code == 0:
            self.final_status = "finished"
        else:
            self.final_status = "error"
            text = "\n".join(self.lines)
            if "out of memory" in text.lower() or "OutOfMemoryError" in text:
                self.error = "Mémoire de la carte graphique insuffisante (6 Go). Baisse la résolution ou la dimension dans « Avancé »."
            else:
                self.error = f"kohya s'est arrêté avec une erreur (code {code}). Regarde le journal ci-dessous."
        if self.on_end:
            try:
                self.on_end(self)
            except Exception as e:  # ne jamais bloquer la fin du job
                self.lines.append(f"[LoRA Creator] erreur en fin de job : {e}")
        self.status = self.final_status  # annoncé seulement une fois les fichiers copiés

    def _line(self, line: str, log):
        m = STEP_RE.search(line)
        if m:
            if self.kind == "train" and "steps:" not in line:
                self.phase = "Génération des images de test"  # barre de la génération, pas de l'entraînement
            else:
                self.step, self.total = int(m.group(2)), int(m.group(3))
                val = float(m.group(6))
                if val > 0:
                    self.rate = val if m.group(7) == "s/it" else 1 / val
                self.eta = m.group(5)
                if m.group(8):
                    self.loss = float(m.group(8))
                if self.kind == "train":
                    if self.first_step_at is None and self.step > 0:
                        self.first_step_at = time.time()
                    if self.step < self.total:
                        self.phase = "Entraînement"
                else:
                    self.phase = "Création des légendes"
            # les barres de progression ne vont pas dans l'historique affiché (trop de lignes)
            if self.lines and STEP_RE.search(self.lines[-1]):
                self.lines[-1] = line.strip()
            else:
                self.lines.append(line.strip())
            return
        log.write(line + "\n")
        log.flush()
        self.lines.append(line.rstrip())
        e = EPOCH_RE.search(line)
        if e:
            self.epoch, self.epochs = int(e.group(1)), int(e.group(2))
            if self.kind == "train":
                self.phase = "Entraînement"
        low = line.lower()
        for key, label in PHASES:
            if key in low:
                self.phase = label
                break

    def cancel(self):
        if self.proc and self.proc.poll() is None:
            self.cancelled = True
            if IS_WIN:
                subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True, **_NO_WINDOW)
            else:
                import signal
                try:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def state(self) -> dict:
        now = self.ended or time.time()
        return {
            "id": self.id, "kind": self.kind, "run_id": self.run_id, "status": self.status, "phase": self.phase,
            "error": self.error, "step": self.step, "total": self.total, "epoch": self.epoch, "epochs": self.epochs,
            "rate": self.rate, "eta": self.eta, "loss": self.loss, "elapsed": round(now - self.started),
            "log": list(self.lines)[-120:],
        }


_job: Job | None = None
_job_lock = threading.Lock()


def current_job() -> Job | None:
    return _job


def _start_job(job: Job):
    global _job
    with _job_lock:
        if _job and _job.status == "running":
            raise RuntimeError("Un travail est déjà en cours sur la carte graphique.")
        _job = job
        job.start()
    return job


# --------------------------------------------------------------------------- légendes automatiques

def start_autotag(slug: str, method: str, overwrite: bool, threshold: float = 0.35) -> Job:
    d = dataset_dir(slug)
    imgs = list_images(d)
    if not imgs:
        raise ValueError("Ajoute des images avant de créer les légendes.")
    ext = ".txt" if overwrite else ".autotag"
    if method == "blip":
        cmd = [str(PYTHON), "finetune/make_captions.py", str(d), "--batch_size", "1", "--caption_extension", ext,
               "--beam_search", "--max_length", "75", "--min_length", "10"]
    else:
        cmd = [str(PYTHON), "finetune/tag_images_by_wd14_tagger.py", str(d), "--onnx", "--repo_id", CFG["tagger_repo"],
               "--model_dir", str(DATA / "models" / "wd14"), "--batch_size", "4", "--thresh", str(threshold),
               "--caption_extension", ext, "--remove_underscore", "--character_tags_first",
               "--max_data_loader_n_workers", "1"]

    def on_end(job: Job):
        if overwrite:
            return
        # sans écrasement : on ne garde la nouvelle légende que si l'image n'en avait pas
        for p in imgs:
            tmp, cap = p.with_suffix(".autotag"), p.with_suffix(".txt")
            if tmp.exists():
                if not cap.exists() or not cap.read_text(encoding="utf-8").strip():
                    tmp.replace(cap)
                else:
                    tmp.unlink()

    return _start_job(Job("tag", cmd, CACHE / "autotag.log", on_end=on_end))


# --------------------------------------------------------------------------- entraînement

def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(p: Path, data) -> None:
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sample_prompts(ltype: str, trigger: str, first_caption: str, family: str, pony: bool, res: int) -> str:
    pre = "score_9, score_8_up, score_7_up, " if pony else ""
    cls = TYPES[ltype]["class"]
    if ltype == "style":
        prompts = [f"{trigger}, a quiet village street at sunset", f"{trigger}, portrait of a woman"]
    elif ltype == "personnage":
        prompts = [f"{trigger}, {first_caption}" if first_caption else f"{trigger}, {cls}",
                   f"{trigger}, {cls}, upper body, smiling, simple background"]
    else:
        prompts = [f"{trigger}, {first_caption}" if first_caption else f"{trigger}, {cls}",
                   f"{trigger}, {cls}, on a wooden table, studio lighting"]
    if family == "sdxl":
        prompts = prompts[:1]
    neg = "lowres, bad anatomy, bad hands, worst quality, low quality, blurry, watermark"
    if pony:
        neg = "score_4, score_5, score_6, " + neg
    return "\n".join(f"{pre}{p} --n {neg} --w {res} --h {res} --s 20 --l 7 --d 42" for p in prompts) + "\n"


def start_training(req: dict) -> Job:
    """req : dataset, name, trigger, type, quality, model (chemin), advanced {...}, auto_forge."""
    models = {m["path"]: m for m in list_models()}
    model = models.get(req["model"])
    if not model:
        raise ValueError("Modèle introuvable ou non reconnu.")
    fam, ltype, quality = model["family"], req["type"], req["quality"]
    if ltype not in TYPES or quality not in QUALITY_EPOCHS:
        raise ValueError("Type ou qualité inconnu.")
    d = dataset_dir(req["dataset"])
    imgs = list_images(d)
    if not imgs:
        raise ValueError("Le dataset ne contient aucune image.")
    name = slugify(req["name"])
    trigger = req["trigger"].strip() or name.lower()

    profile = req.get("profile") if req.get("profile") in PROFILES else DEFAULT_PROFILE
    big = profile == "big"
    p, adv = _params(fam, ltype, quality, profile, req.get("advanced") or {})
    batch = int(p["batch"])
    plan = plan_steps(len(imgs), int(p["target_steps"]), int(p["epochs"]), int(adv.get("repeats") or 0) or None, batch)
    res = int(p["resolution"])

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + name[:30]
    rdir = RUNS / run_id
    out = rdir / "output"
    out.mkdir(parents=True)

    first_cap = ""
    cap0 = imgs[0].with_suffix(".txt")
    if cap0.exists():
        first_cap = ", ".join(cap0.read_text(encoding="utf-8").split(",")[:12]).strip()
    prompts = (req.get("sample_prompts") or "").strip()
    if prompts:
        prompts = "\n".join(
            l if "--w" in l else f"{l} --w {res} --h {res} --s 20 --l 7 --d 42" for l in prompts.splitlines() if l.strip()) + "\n"
    else:
        prompts = _sample_prompts(ltype, trigger, first_cap, fam, model["pony"], res)
    (rdir / "prompts.txt").write_text(prompts, encoding="utf-8")

    subset = {"image_dir": str(d), "num_repeats": plan["repeats"], "caption_extension": ".txt",
              "caption_prefix": f"{trigger},"}
    general = {"enable_bucket": True, "bucket_no_upscale": True, "bucket_reso_steps": 64,
               "min_bucket_reso": 256, "max_bucket_reso": res * 2, "caption_extension": ".txt"}
    if fam == "sd15":
        general.update({"shuffle_caption": True, "keep_tokens": 1})
    dataset_cfg = {"general": general,
                   "datasets": [{"resolution": [res, res], "batch_size": batch, "subsets": [subset]}]}
    (rdir / "dataset.toml").write_text(toml.dumps(dataset_cfg), encoding="utf-8")

    args = {
        "pretrained_model_name_or_path": model["path"],
        "dataset_config": str(rdir / "dataset.toml"),
        "output_dir": str(out),
        "output_name": name,
        "save_model_as": "safetensors",
        "network_module": "networks.lora",
        "network_dim": int(p["dim"]),
        "network_alpha": int(p["alpha"]),
        "learning_rate": float(p["unet_lr"]),
        "unet_lr": float(p["unet_lr"]),
        "optimizer_type": "AdamW8bit",
        "lr_scheduler": "cosine_with_restarts",
        "lr_scheduler_num_cycles": 1,
        "max_train_epochs": plan["epochs"],
        "save_every_n_epochs": plan["save_every"],
        # petite carte : tout pour économiser la VRAM ; gros GPU : bf16 et pas d'économies qui ralentissent
        "mixed_precision": "bf16" if big else "fp16",
        "save_precision": "fp16",
        "gradient_checkpointing": not big,
        ("sdpa" if big else "xformers"): True,
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "seed": int(p["seed"]),
        "max_data_loader_n_workers": 2 if big else 1,
        "persistent_data_loader_workers": True,
        "min_snr_gamma": 5,
        "noise_offset": 0.05 if fam == "sd15" else 0.0357,
        "training_comment": f"LoRA Creator | {ltype} | trigger: {trigger}",
    }
    if fam == "sd15":
        args.update({"clip_skip": 2, "text_encoder_lr": float(p["te_lr"])})
        if float(p["te_lr"]) == 0:
            args["network_train_unet_only"] = True
    else:
        args.update({"network_train_unet_only": True, "cache_text_encoder_outputs": True,
                     "cache_text_encoder_outputs_to_disk": True, "no_half_vae": True})
        if not big:
            args["fp8_base"] = True  # le modèle SDXL ne tient dans 6 Go qu'en fp8
    if p["samples"]:
        args.update({"sample_prompts": str(rdir / "prompts.txt"), "sample_every_n_epochs": plan["save_every"],
                     "sample_sampler": "euler_a"})
    (rdir / "config.toml").write_text(toml.dumps(args), encoding="utf-8")

    run = {
        "id": run_id, "name": name, "trigger": trigger, "type": ltype, "quality": quality,
        "family": fam, "model": model["path"], "model_name": model["name"], "pony": model["pony"],
        "profile": profile, "dataset": req["dataset"], "n_images": len(imgs), "params": {**p, **plan},
        "advanced": adv, "sample_prompts": req.get("sample_prompts") or "", "auto_forge": bool(req.get("auto_forge", True)),
        "estimate": estimate(len(imgs), ltype, quality, {fam: adv}, profile)[fam]["seconds"],
        "status": "running", "started": _now(), "ended": None, "duration": None,
    }
    _write_json(rdir / "run.json", run)

    script = "train_network.py" if fam == "sd15" else "sdxl_train_network.py"
    cmd = [str(PYTHON), script, "--config_file", str(rdir / "config.toml")]
    return _start_job(Job("train", cmd, rdir / "train.log", run_id=run_id, on_end=_finish_training))


def _finish_training(job: Job) -> None:
    rdir = RUNS / job.run_id
    run = _read_json(rdir / "run.json")
    run["status"] = job.final_status
    run["ended"] = _now()
    run["duration"] = round(job.ended - job.started)
    run["error"] = job.error
    if job.final_status == "finished" and job.first_step_at:
        out = rdir / "output"
        n_samples = len(list((out / "sample").glob("*.png"))) if (out / "sample").is_dir() else 0
        steps = max(1, job.total or run["params"]["steps"])
        train_sec = (job.rate or 0) * steps  # compteur de kohya : pas + images de test
        sample_sec = prof(run.get("profile"))["sample_sec"][run["family"]]
        spi = max(0.05, (train_sec - n_samples * sample_sec) / steps)
        run["measured"] = {
            "spi": round(spi, 3),
            "overhead": round(max(0, (job.ended - job.started) - train_sec)),
            "samples": n_samples,
            # les tout petits essais faussent l'estimation
            "reliable": steps * run["params"].get("batch", 1) >= 200,
        }
        final = out / f"{run['name']}.safetensors"
        if final.exists():
            shutil.copy2(final, LORA_OUT / final.name)
            run["final"] = final.name
            if run.get("auto_forge"):
                try:
                    run["forge"] = copy_to_forge(run, final.name)
                except Exception as e:
                    run["forge_error"] = str(e)
    _write_json(rdir / "run.json", run)


def run_files(run_id: str) -> dict:
    rdir = _run_dir(run_id)
    out = rdir / "output"
    loras = sorted((f.name for f in out.glob("*.safetensors")), key=lambda n: (n.count("-"), n)) if out.is_dir() else []
    samples = sorted(((out / "sample").glob("*.png")), key=lambda p: p.stat().st_mtime) if (out / "sample").is_dir() else []
    return {"loras": loras, "samples": [s.name for s in samples]}


def _run_dir(run_id: str) -> Path:
    p = (RUNS / run_id).resolve()
    if p.parent != RUNS.resolve() or not p.is_dir():
        raise ValueError("entraînement introuvable")
    return p


def list_runs() -> list[dict]:
    runs = []
    for d in sorted(RUNS.iterdir(), reverse=True):
        r = _read_json(d / "run.json")
        if r:
            if r.get("status") == "running" and not (_job and _job.run_id == r["id"] and _job.status == "running"):
                r["status"] = "interrupted"
            runs.append(r)
    return runs


def get_run(run_id: str) -> dict:
    r = _read_json(_run_dir(run_id) / "run.json")
    return {**r, **run_files(run_id)}


def copy_to_forge(run: dict, lora_file: str) -> str:
    src = RUNS / run["id"] / "output" / Path(lora_file).name
    if not src.exists():
        raise ValueError("Fichier LoRA introuvable.")
    FORGE_LORA.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    meta_path = FORGE_LORA / f"{stem}.json"
    if (FORGE_LORA / src.name).exists() and "LoRA Creator" not in _read_json(meta_path).get("notes", ""):
        stem += "_dono"  # ne jamais écraser un LoRA qui ne vient pas de LoRA Creator
    shutil.copy2(src, FORGE_LORA / f"{stem}.safetensors")
    _write_json(FORGE_LORA / f"{stem}.json", {
        "description": f"{TYPES[run['type']]['label']} — entraîné sur {run['model_name']}",
        "sd version": "SD1" if run["family"] == "sd15" else "SDXL",
        "activation text": run["trigger"],
        "preferred weight": 0.8,
        "notes": f"Créé avec LoRA Creator le {run.get('ended') or _now()}",
    })
    samples = run_files(run["id"])["samples"]
    if samples:
        shutil.copy2(RUNS / run["id"] / "output" / "sample" / samples[-1], FORGE_LORA / f"{stem}.preview.png")
    return f"{stem}.safetensors"
