"""
mavrec_lib.py — shared code for the MAVREC final-year project.

Single source of truth for: paths, class names, splits, seeds, evaluation,
result records, model adapters (Ultralytics YOLO/RT-DETR, official YOLOv7,
torchvision Faster R-CNN, Hugging Face DETR / Deformable DETR), tiling and
plotting helpers.

Every notebook does:
    import mavrec_lib as ml
    CFG = ml.setup()            # mounts nothing; Drive must already be mounted
and then calls ml.run_experiment(spec) or analysis helpers.

Design rules
------------
* All ground truth lives in canonical COCO JSONs (category ids 1..10 in CLASS_NAMES
  order, boxes in NATIVE image pixels, area = w*h).  APS/APM are therefore always
  measured at native resolution, whatever the network input size was.
* Every model is evaluated by the same pycocotools-based evaluator on predictions
  exported in native coordinates (maxDets = 300 because MAVREC aerial frames carry
  ~78 objects on average; AP@maxDets=100 is also stored for comparison with the paper).
* One JSON record per experiment (no shared CSV that two people write at once).
* Training is resumable; a finished experiment is never retrained unless force=True.
"""
from __future__ import annotations

import os, sys, json, glob, time, math, random, shutil, subprocess, zipfile, re, gc, copy, gzip, threading, datetime
from pathlib import Path

import numpy as np

LIB_VERSION = "1.0.0"

# ----------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------
CLASS_NAMES = ["tram", "bicycle", "van", "truck", "bus",
               "person", "car", "other", "streetlight", "traffic_light"]
NUM_CLASSES = len(CLASS_NAMES)
SEED = 42
# Nested training-subset sizes (500 ⊂ 1000 ⊂ 2000) and ground-view pretraining size.
TRAIN_SIZES = tuple(int(x) for x in os.environ.get("MAVREC_TRAIN_SIZES", "500,1000,2000").split(","))
N_SMALL, N_SCREEN, N_LARGE = TRAIN_SIZES
N_GROUND = int(os.environ.get("MAVREC_N_GROUND", "2000"))

# Size buckets in NATIVE pixels (COCO definition + an extra 'tiny' bucket)
AREA_RANGES = {"all": [0, 1e10], "tiny": [0, 16 ** 2], "small": [0, 32 ** 2],
               "medium": [32 ** 2, 96 ** 2], "large": [96 ** 2, 1e10]}
MAX_DETS = 300

# Default T4-safe batch sizes (per GPU step) by model and input long side.
# Ultralytics & YOLOv7 accumulate gradients to a nominal batch of 64, so the
# effective batch is the same at every resolution.
T4_BATCH = {
    "yolov8m": {640: 16, 960: 8, 1280: 4},
    "yolo11m": {640: 16, 960: 8, 1280: 4},
    "yolov8s": {640: 32, 960: 16, 1280: 8},
    "yolo11s": {640: 32, 960: 16, 1280: 8},
    "rtdetr-l": {640: 8, 960: 4, 1280: 2},
    "yolov7": {640: 8, 960: 4, 1280: 2},
    "frcnn_r50_v1": {640: 8, 960: 4, 1280: 2},
    "frcnn_r50_v2": {640: 8, 960: 4, 1280: 2},
    "frcnn_r101": {640: 6, 960: 3, 1280: 2},
    "detr_r50": {640: 8, 960: 4, 1280: 2},
    "deformable_detr": {640: 4, 960: 2, 1280: 1},
}

# Default epoch budgets per model (fixed budget, final checkpoint is evaluated:
# no early stopping and no checkpoint picking on the validation set).
DEFAULT_EPOCHS = {
    "yolov7": 40, "yolov8m": 40, "yolo11m": 40, "yolov8s": 40, "yolo11s": 40,
    "rtdetr-l": 40, "frcnn_r50_v1": 20, "frcnn_r50_v2": 20, "frcnn_r101": 20,
    "detr_r50": 50, "deformable_detr": 30,
}

MODEL_FAMILY = {
    "yolov7": "yolov7", "yolov8m": "ultralytics", "yolo11m": "ultralytics",
    "yolov8s": "ultralytics", "yolo11s": "ultralytics", "rtdetr-l": "ultralytics",
    "frcnn_r50_v1": "torchvision", "frcnn_r50_v2": "torchvision", "frcnn_r101": "torchvision",
    "detr_r50": "hf", "deformable_detr": "hf",
}
MODEL_GROUP = {  # detector family used in reports
    "yolov7": "YOLO", "yolov8m": "YOLO", "yolo11m": "YOLO", "yolov8s": "YOLO", "yolo11s": "YOLO",
    "rtdetr-l": "Transformer", "detr_r50": "Transformer", "deformable_detr": "Transformer",
    "frcnn_r50_v1": "Two-stage", "frcnn_r50_v2": "Two-stage", "frcnn_r101": "Two-stage",
}
MODEL_BACKBONE = {
    "yolov7": "E-ELAN (YOLOv7)", "yolov8m": "CSP-C2f (m)", "yolo11m": "C3k2+C2PSA (m)",
    "yolov8s": "CSP-C2f (s)", "yolo11s": "C3k2+C2PSA (s)", "rtdetr-l": "HGNetv2-L",
    "frcnn_r50_v1": "ResNet-50-FPN", "frcnn_r50_v2": "ResNet-50-FPN v2",
    "frcnn_r101": "ResNet-101-FPN", "detr_r50": "ResNet-50", "deformable_detr": "ResNet-50 (multi-scale)",
}
PRETRAINED_SOURCE = {
    "yolov7": "yolov7_training.pt (COCO)", "yolov8m": "yolov8m.pt (COCO)", "yolo11m": "yolo11m.pt (COCO)",
    "yolov8s": "yolov8s.pt (COCO)", "yolo11s": "yolo11s.pt (COCO)", "rtdetr-l": "rtdetr-l.pt (COCO)",
    "frcnn_r50_v1": "torchvision COCO_V1", "frcnn_r50_v2": "torchvision COCO_V1 (v2 recipe)",
    "frcnn_r101": "ImageNet R101 backbone + COCO R50-FPN neck/RPN/ROI heads (transplanted)",
    "detr_r50": "facebook/detr-resnet-50 (COCO; 300 queries, query embeddings re-initialised)",
    "deformable_detr": "SenseTime/deformable-detr (COCO)",
}

CFG: dict = {}


# ----------------------------------------------------------------------------------
# Setup / environment
# ----------------------------------------------------------------------------------
def setup(drive_root: str = "/content/drive/MyDrive/MAVREC",
          local_root: str = "/content/mavrec_local",
          smoke: bool | None = None, require_gpu: bool = True, seed: int = SEED, quiet: bool = False):
    """Configure paths, seeds and print an environment report. Call once per notebook."""
    if smoke is None:
        smoke = os.environ.get("MAVREC_SMOKE", "0") == "1"
    CFG.clear()
    CFG.update(dict(
        drive_root=drive_root,
        data_dir=os.path.join(drive_root, "Final_Project_Data"),
        canon_dir=os.path.join(drive_root, "Final_Project_Data", "canonical"),
        packs_dir=os.path.join(drive_root, "Final_Project_Data", "packs"),
        ext_dir=os.path.join(drive_root, "Final_Project_Data", "external"),
        code_dir=os.path.join(drive_root, "Final_Project_Code"),
        results_root=os.path.join(drive_root, "Final_Project_Results", "smoke" if smoke else ""),
        local_root=local_root,
        smoke=smoke, seed=seed,
        smoke_max_res=int(os.environ.get("MAVREC_SMOKE_MAX_RES", "0")) or None,
    ))
    CFG["results_root"] = CFG["results_root"].rstrip("/")
    R = CFG["results_root"]
    CFG.update(dict(
        runs_dir=os.path.join(R, "runs"), records_dir=os.path.join(R, "records"),
        preds_dir=os.path.join(R, "preds"), figures_dir=os.path.join(R, "figures"),
        tables_dir=os.path.join(R, "tables"), decisions_path=os.path.join(R, "decisions.json"),
    ))
    for k in ["data_dir", "canon_dir", "packs_dir", "ext_dir", "code_dir", "runs_dir", "records_dir",
              "preds_dir", "figures_dir", "tables_dir"]:
        os.makedirs(CFG[k], exist_ok=True)
    for sub in ["images", "yolo", "weights", "tmp"]:
        os.makedirs(os.path.join(local_root, sub), exist_ok=True)
    set_seed(seed)
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # YOLOv7 / old checkpoints
    rep = env_report(quiet=quiet)
    CFG["device"] = "cuda" if rep["cuda"] else "cpu"
    if require_gpu and not rep["cuda"]:
        if smoke:
            print("WARNING: no GPU — continuing on CPU because SMOKE mode is on.")
        else:
            raise RuntimeError("No CUDA GPU. In Colab: Runtime > Change runtime type > T4 GPU.")
    if smoke and not quiet:
        print(">>> SMOKE MODE: tiny subsets, 1 epoch; results go to", R)
    return CFG


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _ver(mod):
    try:
        m = __import__(mod)
    except Exception:
        return None
    v = getattr(m, "__version__", None)
    if v is None:
        try:
            from importlib.metadata import version
            v = version(mod)
        except Exception:
            v = "?"
    return v


def env_report(quiet=False):
    import torch
    rep = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.cuda.is_available(),
               torchvision=_ver("torchvision"), ultralytics=_ver("ultralytics"),
               transformers=_ver("transformers"), pycocotools=_ver("pycocotools"),
               numpy=np.__version__, lib=LIB_VERSION)
    if rep["cuda"]:
        p = torch.cuda.get_device_properties(0)
        rep["gpu"] = p.name
        rep["gpu_mem_gb"] = round(p.total_memory / 1024 ** 3, 1)
    if not quiet:
        print("Environment:")
        for k, v in rep.items():
            print(f"  {k:13s}: {v}")
        if rep["cuda"] and "T4" not in rep.get("gpu", ""):
            print("  NOTE: GPU is not a T4. Batch sizes are tuned for 15 GB; results are still valid.")
    return rep


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


class GPUMemPoller:
    """Polls nvidia-smi in a background thread; works for subprocess training too."""

    def __init__(self, interval=5.0):
        self.interval, self.peak_mb, self._stop = interval, None, threading.Event()

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
                if out:
                    v = float(out[0])
                    self.peak_mb = v if self.peak_mb is None else max(self.peak_mb, v)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        if shutil.which("nvidia-smi"):
            self._t = threading.Thread(target=self._poll, daemon=True)
            self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()

    @property
    def peak_gb(self):
        return None if self.peak_mb is None else round(self.peak_mb / 1024, 2)


# ----------------------------------------------------------------------------------
# Small IO helpers
# ----------------------------------------------------------------------------------
def load_json(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def save_json(obj, path, indent=None):
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    tmp = str(path) + ".tmp"
    if str(path).endswith(".gz"):
        with gzip.open(tmp, "wt") as f:
            json.dump(obj, f)
    else:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=indent, default=_json_default)
    os.replace(tmp, path)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def sequence_id(file_name: str) -> str:
    """'scene_10_sdu_30Sec_droneView_3_000026.jpg' -> 'scene_10_sdu_30Sec_droneView_3'."""
    stem = Path(file_name).stem
    m = re.match(r"^(.*?)[_\-]?(\d+)$", stem)
    return m.group(1) if m and m.group(1) else stem

# ----------------------------------------------------------------------------------
# Raw data discovery (used by notebook 00)
# ----------------------------------------------------------------------------------
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
_SKIP_DIRS = ("Final_Project_Data", "Final_Project_Results", "Final_Project_Code")


class SourceIndex:
    """Maps image basename -> where the bytes live (loose file or member of a zip).

    Works with MAVREC zips kept on Google Drive (images are extracted on demand,
    so the 20+ GB training archive never has to be fully unpacked)."""

    def __init__(self):
        self.files = {}      # basename -> ("file", path) | ("zip", zip_path, member)
        self.dupes = 0
        self._local = threading.local()   # one ZipFile handle per thread (parallel pack building)
        self._open = []                    # every handle opened (closed before a Drive cache recycle)
        self._lock = threading.Lock()
        self.offset = {}                   # basename -> byte offset inside its zip (read in on-disk order)

    def add_dir(self, root):
        n = 0
        for base, dirs, files in os.walk(root):
            if any(s in base for s in _SKIP_DIRS):
                continue
            for f in files:
                if f.lower().endswith(IMG_EXT):
                    self._add(f, ("file", os.path.join(base, f)))
                    n += 1
        return n

    def add_zip(self, zpath):
        n = 0
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                m = info.filename
                if m.lower().endswith(IMG_EXT) and not m.startswith("__MACOSX"):
                    b = os.path.basename(m)
                    if b not in self.files:
                        self.offset[b] = info.header_offset
                    self._add(b, ("zip", zpath, m))
                    n += 1
        return n

    def _zip(self, zpath):
        zs = getattr(self._local, "zips", None)
        if zs is None:
            zs = self._local.zips = {}
        if zpath not in zs or zs[zpath].fp is None:
            zs[zpath] = zipfile.ZipFile(zpath)
            with self._lock:
                self._open.append(zs[zpath])
        return zs[zpath]

    def close(self):
        """Close all zip handles of all threads (they reopen lazily on the next read)."""
        with self._lock:
            for z in self._open:
                try:
                    z.close()
                except Exception:
                    pass
            self._open = []

    def _add(self, key, val):
        if key in self.files:
            self.dupes += 1  # same basename seen twice: keep the first occurrence
            return
        self.files[key] = val

    def has(self, basename):
        return basename in self.files

    def read_bytes(self, basename):
        src = self.files[basename]
        if src[0] == "file":
            with open(src[1], "rb") as f:
                return f.read()
        return self._zip(src[1]).read(src[2])

    def image_size(self, basename):
        """(width, height) from the file header (no full decode)."""
        from PIL import Image
        src = self.files[basename]
        if src[0] == "file":
            with Image.open(src[1]) as im:
                return im.size
        with self._zip(src[1]).open(src[2]) as f, Image.open(f) as im:
            return im.size

    def read_image(self, basename):
        import cv2
        buf = np.frombuffer(self.read_bytes(basename), np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if im is None:
            raise IOError(f"cannot decode {basename}")
        return im  # BGR


def find_raw_sources(root=None, extra_dirs=()):
    """List zip files and loose COCO JSONs under the Drive MAVREC folder (and extra dirs)."""
    root = root or CFG["drive_root"]
    zips, jsons = [], []
    for d in [root, *extra_dirs]:
        if not d or not os.path.exists(d):
            continue
        for base, dirs, files in os.walk(d):
            if any(s in base for s in _SKIP_DIRS):
                continue
            for f in files:
                p = os.path.join(base, f)
                if f.lower().endswith(".zip"):
                    zips.append(p)
                elif f.lower().endswith(".json"):
                    jsons.append(p)
    return sorted(set(zips)), sorted(set(jsons))


def extract_jsons_from_zips(zips, dest=None):
    dest = dest or os.path.join(CFG["local_root"], "raw_json")
    out = []
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            for m in z.namelist():
                if m.lower().endswith(".json") and not m.startswith("__MACOSX"):
                    tgt = os.path.join(dest, Path(zp).stem, m)
                    if not os.path.exists(tgt):
                        os.makedirs(os.path.dirname(tgt), exist_ok=True)
                        with z.open(m) as src, open(tgt, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                    out.append(tgt)
    return out


def _guess(path, data, key):
    toks = norm_name(path)
    names = " ".join(norm_name(im.get("file_name", "")) for im in data.get("images", [])[:50])
    if key == "view":
        if "ground" in toks or "ground" in names or "gopro" in names or "iphone" in names:
            if not ("aerial" in toks or "drone" in toks):
                return "ground"
        if "aerial" in toks or "drone" in toks or "drone" in names or "aerial" in names:
            return "aerial"
        return "?"
    parts = [p.lower() for p in Path(path).parts]
    alias = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
    for part in [Path(path).stem.lower()] + parts[::-1]:
        for tok, s in alias.items():
            if re.search(rf"(^|[^a-z]){tok}($|[^a-z])", part):
                return s
    return "?"


def inspect_annotation_files(json_paths):
    """Return a DataFrame describing every COCO-like JSON found."""
    import pandas as pd
    rows = []
    for p in json_paths:
        try:
            d = load_json(p)
        except Exception as e:
            continue
        if not isinstance(d, dict) or "images" not in d or "annotations" not in d:
            continue
        rows.append(dict(path=p, images=len(d["images"]), annotations=len(d["annotations"]),
                         categories=[c.get("name") for c in d.get("categories", [])],
                         view=_guess(p, d, "view"), split=_guess(p, d, "split")))
    return pd.DataFrame(rows)


def category_mapping(categories, class_names=CLASS_NAMES):
    """Map source category ids -> canonical ids 1..N by (normalised) name. Fails loudly."""
    want = {norm_name(n): i + 1 for i, n in enumerate(class_names)}
    alias = {"streetlamp": "streetlight", "pedestrian": "person", "people": "person"}
    m, unmatched = {}, []
    for c in categories:
        k = norm_name(c["name"])
        k = alias.get(k, k)
        if k in want:
            m[c["id"]] = want[k]
        else:
            unmatched.append(c["name"])
    missing = [n for n in class_names if want[norm_name(n)] not in m.values()]
    return m, unmatched, missing


def canonicalize(src_json, view, split, source_index=None, class_names=CLASS_NAMES, check_sizes=True):
    """Build a canonical COCO dict: ids 1..N in class_names order, file_name='<split>/<stem>.jpg' (packs are JPEG),
    src_name = original basename, boxes clipped to the image, area=w*h, iscrowd=0, plus 'seq' (video sequence id).

    Real MAVREC release: frames are a MIX of 3840x2160 and 2704x1520, and the official test JSON declares every
    image as 600x337 with boxes in that space. So the declared size is checked against the real file once per
    (sequence, declared size) group (first and last image), and boxes are rescaled per axis when they differ."""
    d = load_json(src_json) if isinstance(src_json, str) else src_json
    cmap, unmatched, missing = category_mapping(d["categories"], class_names)
    if unmatched:
        print(f"  [{view}/{split}] categories ignored (not in CLASS_NAMES): {unmatched}")
    if missing:
        print(f"  [{view}/{split}] WARNING classes absent from this file: {missing}")
    images, id_map, n_missing = [], {}, 0
    for im in sorted(d["images"], key=lambda x: os.path.basename(x["file_name"])):
        b = os.path.basename(im["file_name"])
        if source_index is not None and not source_index.has(b):
            n_missing += 1
            continue
        new_id = len(images) + 1
        id_map[im["id"]] = new_id
        images.append(dict(id=new_id, file_name=f"{split}/{Path(b).stem}.jpg",
                           width=int(round(float(im["width"]))), height=int(round(float(im["height"]))),
                           seq=sequence_id(b), src_name=b))
    if n_missing:
        print(f"  [{view}/{split}] {n_missing} images listed in the JSON were not found in the image sources")
    # declared size vs real file size, per (sequence, declared size) group
    scale = {}  # canonical image id -> (sx, sy)
    if source_index is not None and check_sizes and images:
        groups = {}
        for im in images:
            groups.setdefault((im["seq"], im["width"], im["height"]), []).append(im)
        fixed = {}
        for (seq, dw, dh), ims in groups.items():
            probes = {source_index.image_size(i["src_name"]) for i in {id(ims[0]): ims[0], id(ims[-1]): ims[-1]}.values()}
            if len(probes) > 1:  # sizes vary inside the group: probe every image
                sizes = {i["id"]: source_index.image_size(i["src_name"]) for i in ims}
            else:
                sz = probes.pop()
                sizes = {i["id"]: sz for i in ims}
            for i in ims:
                fw, fh = sizes[i["id"]]
                if (fw, fh) != (dw, dh):
                    scale[i["id"]] = (fw / dw, fh / dh)
                    fixed[(dw, dh, fw, fh)] = fixed.get((dw, dh, fw, fh), 0) + 1
                    i["width"], i["height"] = fw, fh
        for (dw, dh, fw, fh), n in sorted(fixed.items()):
            print(f"  [{view}/{split}] {n} images declared {dw}x{dh} but are {fw}x{fh}: boxes rescaled "
                  f"x{fw / dw:.4f} / x{fh / dh:.4f}")
    anns, n_drop = [], 0
    for a in d["annotations"]:
        if a["image_id"] not in id_map or a["category_id"] not in cmap:
            continue
        iid = id_map[a["image_id"]]
        im = images[iid - 1]
        sx, sy = scale.get(iid, (1.0, 1.0))
        x, y, w, h = [float(v) for v in a["bbox"]]
        x, y, w, h = x * sx, y * sy, w * sx, h * sy
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(im["width"]), x + w), min(float(im["height"]), y + h)
        w, h = x2 - x1, y2 - y1
        if w < 1 or h < 1:
            n_drop += 1
            continue
        anns.append(dict(id=len(anns) + 1, image_id=iid, category_id=cmap[a["category_id"]],
                         bbox=[round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)], area=round(w * h, 2), iscrowd=0))
    if n_drop:
        print(f"  [{view}/{split}] {n_drop} boxes dropped (< 1 px after clipping to the image)")
    cats = [dict(id=i + 1, name=n, supercategory="object") for i, n in enumerate(class_names)]
    return dict(info=dict(view=view, split=split, source=str(src_json), created=now(), lib=LIB_VERSION),
                images=images, annotations=anns, categories=cats)


def subset_coco(d, image_ids, name=None):
    keep = set(image_ids)
    out = dict(info=dict(d.get("info", {}), subset=name), categories=d["categories"],
               images=[im for im in d["images"] if im["id"] in keep],
               annotations=[a for a in d["annotations"] if a["image_id"] in keep])
    return out


def sequence_balanced_order(images, seed=SEED):
    """Deterministic ordering that spreads picks across video sequences.
    Prefixes of this order give NESTED subsets (500 ⊂ 1000 ⊂ 2000) that are not just the
    first frames of the first clips (Review 2 used zip order, i.e. a handful of clips)."""
    rng = random.Random(seed)
    by_seq = {}
    for im in sorted(images, key=lambda x: x["file_name"]):
        by_seq.setdefault(im.get("seq") or sequence_id(im["file_name"]), []).append(im)
    seqs = sorted(by_seq)
    rng.shuffle(seqs)
    for s in seqs:
        rng.shuffle(by_seq[s])
    order, depth = [], 0
    while True:
        added = False
        for s in seqs:
            if depth < len(by_seq[s]):
                order.append(by_seq[s][depth])
                added = True
        if not added:
            break
        depth += 1
    return order


def make_nested_subsets(d, sizes=(500, 1000, 2000), seed=SEED):
    order = sequence_balanced_order(d["images"], seed)
    out = {}
    for s in sizes:
        if s > len(order):
            print(f"  only {len(order)} images available; subset {s} skipped")
            continue
        out[s] = subset_coco(d, [im["id"] for im in order[:s]], name=f"n{s}")
    ks = sorted(out)
    for a, b in zip(ks, ks[1:]):
        A = {im["src_name"] for im in out[a]["images"]}
        B = {im["src_name"] for im in out[b]["images"]}
        assert A <= B, "subsets are not nested"
    return out


def split_by_sequence(d, frac=0.5, seed=SEED):
    """Fallback when no official test JSON exists: split val sequences into val/test halves."""
    rng = random.Random(seed)
    seqs = sorted({im["seq"] for im in d["images"]})
    rng.shuffle(seqs)
    k = max(1, int(round(len(seqs) * frac)))
    a = {s for s in seqs[:k]}
    ids_a = [im["id"] for im in d["images"] if im["seq"] in a]
    ids_b = [im["id"] for im in d["images"] if im["seq"] not in a]
    return subset_coco(d, ids_a, "val_half"), subset_coco(d, ids_b, "test_half")


def leakage_report(named):
    """named: {name: coco_dict}. Reports image and sequence overlap between splits."""
    import pandas as pd
    rows = []
    names = list(named)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = named[names[i]], named[names[j]]
            ia = {im["src_name"] for im in A["images"]}
            ib = {im["src_name"] for im in B["images"]}
            sa = {im["seq"] for im in A["images"]}
            sb = {im["seq"] for im in B["images"]}
            rows.append(dict(a=names[i], b=names[j], shared_images=len(ia & ib), shared_sequences=len(sa & sb)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------
# Image packs (resized caches stored as zips on Drive, unpacked to local disk)
# ----------------------------------------------------------------------------------
def pack_dir(pack):
    return os.path.join(CFG["local_root"], "images", pack)


def pack_zip(pack):
    return os.path.join(CFG["packs_dir"], f"{pack}.zip")


def free_disk_gb(path="/"):
    return shutil.disk_usage(path).free / 1e9


def recycle_drive_cache(mountpoint="/content/drive"):
    """Colab's Google Drive client caches every byte read through the mount on the local disk and does not evict it
    fast enough: reading ~4k frames from the 102 GB MAVREC train.zip filled the 113 GB VM (62 GB content_cache,
    'No space left on device'). flush_and_unmount() + mount() empties the cache (no new consent prompt within the
    same VM). Returns True if a recycle happened."""
    try:
        from google.colab import drive
    except ImportError:
        return False
    before = free_disk_gb()
    drive.flush_and_unmount()
    drive.mount(mountpoint)
    print(f"  Drive cache recycled: free disk {before:.1f} -> {free_disk_gb():.1f} GB")
    return True


def build_packs(packs, coco_dicts, source_index, quality=95, upload=True, workers=6, chunk=300, min_free_gb=15,
                recycle=recycle_drive_cache):
    """Write the images of the given canonical dicts into several packs in ONE pass.
    packs: list of (pack_name, long_side) — long_side=None keeps the native size. Each source image is read and
    decoded once for all packs (the real MAVREC frames are 5-12 MB PNGs inside a 102 GB zip on Drive, where reading
    and decoding dominate), reads run in a thread pool (Drive latency) in on-disk order, files are written atomically.
    Output: LOCAL/images/<pack>/<file_name> as JPEG (JPEG sources copied as-is at native size). Then zipped to Drive."""
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    todo = {}
    for d in coco_dicts:
        for im in d["images"]:
            todo[im["file_name"]] = im["src_name"]
    outs = {p: pack_dir(p) for p, _ in packs}
    # re-running notebook 00 in a new runtime: the packs are already on Drive -> unpack them instead of rebuilding
    if upload and all(os.path.exists(pack_zip(p)) for p, _ in packs):
        n_ok = []
        for p, _ in packs:
            ensure_pack(p)
            have = {os.path.relpath(os.path.join(b_, f), outs[p]) for b_, _, fs in os.walk(outs[p]) for f in fs}
            n_ok.append(all(fn in have for fn in todo))
        if all(n_ok):
            print(f"  [{', '.join(p for p, _ in packs)}] already on Drive and complete ({len(todo)} images) — reused")
            return [outs[p] for p, _ in packs]
        print("  packs on Drive are incomplete for this dataset -> building the missing images")

    def complete(path):
        """Existing file that is a whole JPEG (ends with the EOI marker). An earlier run killed mid-write
        (Colab disconnect / interrupt) can leave a truncated file, which is rebuilt."""
        try:
            with open(path, "rb") as f:
                f.seek(-2, os.SEEK_END)
                return f.read(2) == b"\xff\xd9" or not path.lower().endswith((".jpg", ".jpeg"))
        except OSError:
            return False

    def missing(fn):
        return [(p, ls) for p, ls in packs if not complete(os.path.join(outs[p], fn))]

    jobs = [(fn, src, missing(fn)) for fn, src in todo.items()]
    jobs = [j for j in jobs if j[2]]
    jobs.sort(key=lambda j: (str(source_index.files[j[1]][1]), source_index.offset.get(j[1], 0)))

    def work(job):
        fn, src, need = job
        raw = source_index.read_bytes(src)
        img = None
        for p, ls in need:
            tgt = os.path.join(outs[p], fn)
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            tmp = tgt + ".part"
            if ls is None and src.lower().endswith((".jpg", ".jpeg")) and tgt.lower().endswith((".jpg", ".jpeg")):
                with open(tmp, "wb") as f:  # already JPEG: keep the original bytes
                    f.write(raw)
            else:  # PNG sources (real MAVREC) are re-encoded as JPEG q95: ~5x smaller packs on Drive
                if img is None:
                    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        raise IOError(f"cannot decode {src}")
                h, w = img.shape[:2]
                sc = (ls / max(h, w)) if ls else 1.0
                out = cv2.resize(img, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else img
                ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not ok:
                    raise IOError(f"could not encode {tgt}")
                with open(tmp, "wb") as f:
                    f.write(buf.tobytes())
            os.replace(tmp, tgt)
        return fn

    names = ", ".join(p for p, _ in packs)
    t0, n = time.time(), 0
    for c0 in range(0, len(jobs), chunk):
        # between chunks no read is in flight: safe point to empty Colab's Drive cache if the disk is filling up
        if c0 and free_disk_gb() < min_free_gb:
            source_index.close()
            recycle()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for _ in ex.map(work, jobs[c0:c0 + chunk]):
                n += 1
                if n % 250 == 0:
                    el = time.time() - t0
                    print(f"  [{names}] {n}/{len(jobs)} images ({el:.0f}s, ~{el / n * (len(jobs) - n) / 60:.0f} min left, "
                          f"{free_disk_gb():.0f} GB free)")
    for p, _ in packs:
        Path(outs[p], ".done").write_text(now())
    print(f"  [{names}] {len(todo)} images ready locally ({n} new) in {time.time() - t0:.0f}s")
    if upload:
        for p, _ in packs:
            _zip_dir(outs[p], pack_zip(p))
    return [outs[p] for p, _ in packs]


def build_pack(pack, coco_dicts, source_index, long_side=None, quality=95, upload=True):
    """Single-pack wrapper around build_packs()."""
    return build_packs([(pack, long_side)], coco_dicts, source_index, quality=quality, upload=upload)[0]


def _zip_dir(src_dir, zpath):
    tmp = os.path.join(CFG["local_root"], "tmp", os.path.basename(zpath))
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for base, dirs, files in os.walk(src_dir):
            for f in files:
                p = os.path.join(base, f)
                z.write(p, os.path.relpath(p, src_dir))
    os.makedirs(os.path.dirname(zpath), exist_ok=True)
    shutil.copy(tmp, zpath)
    os.remove(tmp)
    print(f"  uploaded {zpath} ({os.path.getsize(zpath) / 1e9:.2f} GB)")


def ensure_pack(pack):
    """Make sure LOCAL/images/<pack> exists (copies + unzips from Drive once per session)."""
    d = pack_dir(pack)
    if os.path.exists(os.path.join(d, ".done")):
        return d
    z = pack_zip(pack)
    if not os.path.exists(z):
        raise FileNotFoundError(f"Pack {z} not found. Run notebook 00 (or the tiling builder) first.")
    t0 = time.time()
    tmp = os.path.join(CFG["local_root"], "tmp", os.path.basename(z))
    shutil.copy(z, tmp)
    with zipfile.ZipFile(tmp) as zz:
        zz.extractall(d)
    os.remove(tmp)
    Path(d, ".done").write_text(now())
    print(f"Pack '{pack}' unpacked to local disk in {time.time() - t0:.0f}s")
    return d


def image_path(pack, file_name):
    return os.path.join(pack_dir(pack), file_name)


def read_rgb(path):
    import cv2
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise IOError(f"cannot read {path}")
    return np.ascontiguousarray(im[:, :, ::-1])


# ----------------------------------------------------------------------------------
# Canonical JSON names
# ----------------------------------------------------------------------------------
def json_path(name):
    """Resolve a dataset name ('aerial_train_1000', 'aerial_val', 'ext/visdrone_val', ...) or a path."""
    if os.path.isabs(name) or name.endswith(".json"):
        p = name
    elif name.startswith("ext/"):
        p = os.path.join(CFG["ext_dir"], name[4:] + ".json")
    else:
        p = os.path.join(CFG["canon_dir"], name + ".json")
    if CFG.get("smoke"):
        return _smoke_json(p)
    return p


def _smoke_json(p):
    """In smoke mode use the first few images of every dataset."""
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    tgt = os.path.join(CFG["local_root"], "smoke_json", os.path.basename(p))
    if not os.path.exists(tgt) or os.path.getmtime(tgt) < os.path.getmtime(p):
        d = load_json(p)
        n = 8 if any(k in os.path.basename(p) for k in ["val", "test"]) else 16
        save_json(subset_coco(d, [im["id"] for im in d["images"][:n]], "smoke"), tgt)
    return tgt


def load_dataset(name):
    return load_json(json_path(name))


# ----------------------------------------------------------------------------------
# Dataset statistics & plots
# ----------------------------------------------------------------------------------
def dataset_stats(d):
    import pandas as pd
    names = {c["id"]: c["name"] for c in d["categories"]}
    areas = np.array([a["area"] for a in d["annotations"]]) if d["annotations"] else np.zeros(0)
    per_img = pd.Series([a["image_id"] for a in d["annotations"]]).value_counts()
    cls = pd.Series([names[a["category_id"]] for a in d["annotations"]]).value_counts()
    return dict(
        images=len(d["images"]), annotations=len(d["annotations"]),
        sequences=len({im.get("seq") for im in d["images"]}),
        objects_per_image_mean=float(per_img.mean()) if len(per_img) else 0.0,
        objects_per_image_max=int(per_img.max()) if len(per_img) else 0,
        frac_images_over_100_objects=float((per_img > 100).mean()) if len(per_img) else 0.0,
        tiny_pct=float((areas < 16 ** 2).mean() * 100) if len(areas) else 0.0,
        small_pct=float((areas < 32 ** 2).mean() * 100) if len(areas) else 0.0,
        medium_pct=float(((areas >= 32 ** 2) & (areas < 96 ** 2)).mean() * 100) if len(areas) else 0.0,
        large_pct=float((areas >= 96 ** 2).mean() * 100) if len(areas) else 0.0,
        median_box_side_px=float(np.median(np.sqrt(areas))) if len(areas) else 0.0,
        class_counts=cls.to_dict(),
    )


def plot_dataset_overview(named, out_path=None):
    import matplotlib.pyplot as plt
    import pandas as pd
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    cc = pd.DataFrame({k: pd.Series(dataset_stats(d)["class_counts"]) for k, d in named.items()}).reindex(CLASS_NAMES).fillna(0)
    (cc / cc.sum()).plot.bar(ax=axes[0])
    axes[0].set_title("Class distribution (fraction of boxes)")
    axes[0].set_yscale("log")
    for k, d in named.items():
        s = np.sqrt([a["area"] for a in d["annotations"]])
        axes[1].hist(s, bins=np.logspace(0, 3, 50), histtype="step", density=True, label=k)
    for v, ls in [(16, ":"), (32, "--"), (96, "-.")]:
        axes[1].axvline(v, color="k", ls=ls, lw=1)
    axes[1].set_xscale("log")
    axes[1].set_title("Box size sqrt(area) at native resolution (16/32/96 px)")
    axes[1].legend(fontsize=8)
    for k, d in named.items():
        per = pd.Series([a["image_id"] for a in d["annotations"]]).value_counts()
        axes[2].hist(per, bins=40, histtype="step", label=k)
    axes[2].axvline(100, color="r", ls="--", lw=1)
    axes[2].set_title("Objects per image (red: COCO maxDets=100)")
    axes[2].legend(fontsize=8)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


def effective_size_table(d, resolutions=(640, 960, 1280)):
    """How big (median / % under 8 px) objects become after resizing to each input size."""
    import pandas as pd
    long_side = {im["id"]: max(im["width"], im["height"]) for im in d["images"]}  # frames have mixed sizes
    L = np.array([long_side[a["image_id"]] for a in d["annotations"]], float)
    sides = np.sqrt([a["area"] for a in d["annotations"]])
    rows = []
    for r in list(resolutions) + ["native"]:
        sc = np.ones_like(L) if r == "native" else r / L
        e = sides * sc
        rows.append(dict(input_long_side=r, scale=round(float(np.median(sc)), 3), median_object_px=round(float(np.median(e)), 1),
                         pct_under_8px=round(float((e < 8).mean() * 100), 1),
                         pct_under_16px=round(float((e < 16).mean() * 100), 1)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------
# YOLO-format datasets (Ultralytics & YOLOv7) built from canonical JSONs via symlinks
# ----------------------------------------------------------------------------------
def make_yolo_dataset(key, train_name, val_name, pack, class_names=CLASS_NAMES, val_pack=None):
    """Creates LOCAL/yolo/<key>/{images,labels}/{train,val} (+ data.yaml) and returns the yaml path.
    val_pack: pack holding the validation images (defaults to `pack`). Needed for tile training, where the
    training pack contains only tiles and the full validation frames live in the eval pack."""
    val_pack = val_pack or pack
    import yaml
    root = os.path.join(CFG["local_root"], "yolo", key + ("_smoke" if CFG.get("smoke") else ""))
    yml = os.path.join(root, "data.yaml")
    stamp = os.path.join(root, ".stamp")
    sig = json.dumps([json_path(train_name), json_path(val_name), pack, val_pack])
    if os.path.exists(stamp) and Path(stamp).read_text() == sig:
        return yml
    shutil.rmtree(root, ignore_errors=True)
    ensure_pack(pack)
    ensure_pack(val_pack)
    for split, name, pk in [("train", train_name, pack), ("val", val_name, val_pack)]:
        d = load_json(json_path(name))
        idir, ldir = os.path.join(root, "images", split), os.path.join(root, "labels", split)
        os.makedirs(idir, exist_ok=True)
        os.makedirs(ldir, exist_ok=True)
        by = {}
        for a in d["annotations"]:
            by.setdefault(a["image_id"], []).append(a)
        for im in d["images"]:
            src = image_path(pk, im["file_name"])
            flat = im["file_name"].replace("/", "__")
            os.symlink(src, os.path.join(idir, flat))
            W, H = float(im["width"]), float(im["height"])
            lines = []
            for a in by.get(im["id"], []):
                x, y, w, h = a["bbox"]
                lines.append(f"{a['category_id'] - 1} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
            Path(ldir, Path(flat).stem + ".txt").write_text("\n".join(lines))
    cfg = dict(path=root, train="images/train", val="images/val", nc=len(class_names), names=list(class_names))
    with open(yml, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    # YOLOv7 wants absolute train/val paths and no 'path' key
    cfg7 = dict(train=os.path.join(root, "images", "train"), val=os.path.join(root, "images", "val"),
                nc=len(class_names), names=list(class_names))
    with open(os.path.join(root, "data_yolov7.yaml"), "w") as f:
        yaml.safe_dump(cfg7, f, sort_keys=False)
    Path(stamp).write_text(sig)
    return yml


# ----------------------------------------------------------------------------------
# Tiling: training tiles from native-resolution frames
# ----------------------------------------------------------------------------------
def tile_starts(L, t, stride):
    if L <= t:
        return [0]
    s = list(range(0, L - t + 1, stride))
    if s[-1] + t < L:
        s.append(L - t)
    return s


def tile_grid(W, H, tile=640, overlap=0.2):
    stride = max(1, int(round(tile * (1 - overlap))))
    return [(x, y, min(tile, W), min(tile, H)) for y in tile_starts(H, tile, stride) for x in tile_starts(W, tile, stride)]


def build_tile_dataset(train_name, tile=640, overlap=0.2, min_visible=0.3, keep_empty=0.1,
                       native_pack="aerial_native", seed=SEED):
    """Cuts every native frame of `train_name` into overlapping tiles; boxes are clipped and kept
    if >= min_visible of their area is inside the tile. Returns (json_name, pack_name).
    The pack is cached on Drive so it is generated only once per subset."""
    import cv2
    tag = f"tiles_{train_name}_t{tile}_o{int(overlap * 100)}"
    if CFG.get("smoke"):
        tag += "_smoke"
    jpath = os.path.join(CFG["canon_dir"], "derived", tag + ".json")
    if os.path.exists(jpath) and (os.path.exists(pack_zip(tag)) or os.path.exists(os.path.join(pack_dir(tag), ".done"))):
        ensure_pack(tag) if not os.path.exists(os.path.join(pack_dir(tag), ".done")) else None
        return jpath, tag
    ensure_pack(native_pack)
    d = load_json(json_path(train_name))
    rng = random.Random(seed)
    by = {}
    for a in d["annotations"]:
        by.setdefault(a["image_id"], []).append(a)
    images, anns, out = [], [], pack_dir(tag)
    t0 = time.time()
    for k, im in enumerate(d["images"]):
        arr = cv2.imread(image_path(native_pack, im["file_name"]))
        H, W = arr.shape[:2]
        for (tx, ty, tw, th) in tile_grid(W, H, tile, overlap):
            boxes = []
            for a in by.get(im["id"], []):
                x, y, w, h = a["bbox"]
                x1, y1, x2, y2 = max(x, tx), max(y, ty), min(x + w, tx + tw), min(y + h, ty + th)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                if (x2 - x1) * (y2 - y1) < min_visible * w * h:
                    continue
                boxes.append((a["category_id"], [x1 - tx, y1 - ty, x2 - x1, y2 - y1]))
            if not boxes and rng.random() > keep_empty:
                continue
            fn = f"{Path(im['file_name']).stem}_x{tx}_y{ty}.jpg"
            tgt = os.path.join(out, "tiles", fn)
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            cv2.imwrite(tgt, arr[ty:ty + th, tx:tx + tw], [cv2.IMWRITE_JPEG_QUALITY, 95])
            tid = len(images) + 1
            images.append(dict(id=tid, file_name=f"tiles/{fn}", width=tw, height=th, seq=im.get("seq"),
                               src_name=fn, parent=im["file_name"], offset=[tx, ty]))
            for c, b in boxes:
                anns.append(dict(id=len(anns) + 1, image_id=tid, category_id=c, bbox=[round(v, 2) for v in b],
                                 area=round(b[2] * b[3], 2), iscrowd=0))
        if (k + 1) % 200 == 0:
            print(f"  tiled {k + 1}/{len(d['images'])} frames, {len(images)} tiles ({time.time() - t0:.0f}s)")
    Path(out, ".done").write_text(now())
    save_json(dict(info=dict(tiles_of=train_name, tile=tile, overlap=overlap, min_visible=min_visible),
                   images=images, annotations=anns, categories=d["categories"]), jpath)
    print(f"Tile dataset {tag}: {len(images)} tiles, {len(anns)} boxes from {len(d['images'])} frames")
    if not CFG.get("smoke"):
        _zip_dir(out, pack_zip(tag))
    return jpath, tag

# ----------------------------------------------------------------------------------
# Unified evaluation (pycocotools) — identical for every model family
# ----------------------------------------------------------------------------------
def _quiet_coco(gt_path_or_dict):
    from pycocotools.coco import COCO
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        if isinstance(gt_path_or_dict, dict):
            c = COCO()
            c.dataset = copy.deepcopy(gt_path_or_dict)
            c.createIndex()
        else:
            c = COCO(gt_path_or_dict)
    return c


def coco_evaluate(gt, dets, cat_ids=None, max_dets=MAX_DETS):
    """gt: path or dict (canonical COCO). dets: list of {image_id, category_id, bbox[xywh], score}
    in NATIVE pixels. Returns metrics in percent (AP points)."""
    from pycocotools.cocoeval import COCOeval
    import contextlib, io
    cgt = _quiet_coco(gt)
    keys = ["mAP50_95", "mAP50", "AP75", "APt", "APS", "APM", "APL", "AR300", "mAP50_95_md100", "APS_md100"]
    if not dets:
        return {k: 0.0 for k in keys} | {"per_class_AP": {}, "per_class_AP50": {}}
    with contextlib.redirect_stdout(io.StringIO()):
        cdt = cgt.loadRes(copy.deepcopy(dets))
        E = COCOeval(cgt, cdt, "bbox")
        E.params.maxDets = [1, 100, max_dets]
        E.params.areaRng = [AREA_RANGES[k] for k in ["all", "tiny", "small", "medium", "large"]]
        E.params.areaRngLbl = ["all", "tiny", "small", "medium", "large"]
        if cat_ids:
            E.params.catIds = list(cat_ids)
        E.evaluate()
        E.accumulate()
    P = E.eval["precision"]  # [T, R, K, A, M]
    Rr = E.eval["recall"]    # [T, K, A, M]

    def ap(t=None, a=0, m=2, k=None):
        p = P[:, :, :, a, m] if t is None else P[[t], :, :, a, m]
        if k is not None:
            p = p[:, :, [k]]
        p = p[p > -1]
        return float(p.mean() * 100) if p.size else float("nan")

    r = Rr[:, :, 0, 2]
    out = dict(mAP50_95=ap(), mAP50=ap(t=0), AP75=ap(t=5), APt=ap(a=1), APS=ap(a=2), APM=ap(a=3), APL=ap(a=4),
               AR300=float(r[r > -1].mean() * 100) if (r > -1).any() else float("nan"),
               mAP50_95_md100=ap(m=1), APS_md100=ap(a=2, m=1))
    names = {c["id"]: c["name"] for c in cgt.dataset["categories"]}
    out["per_class_AP"] = {names[c]: ap(k=i) for i, c in enumerate(E.params.catIds)}
    out["per_class_AP50"] = {names[c]: ap(t=0, k=i) for i, c in enumerate(E.params.catIds)}
    out = {k: (round(v, 3) if isinstance(v, float) and not math.isnan(v) else (None if isinstance(v, float) else v))
           for k, v in out.items()}
    return out


def _iou_matrix(a, b):
    """a: [N,4] xywh, b: [M,4] xywh"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, float); b = np.asarray(b, float)
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1)[None] - inter + 1e-9)


def match_detections(gt, dets, iou_thr=0.5, class_aware=True):
    """Greedy score-ordered matching per image (per class if class_aware).
    Returns arrays (scores, tp_flags) for dets and per-GT best matched score (or -1) + matched pred class."""
    gtd = gt if isinstance(gt, dict) else load_json(gt)
    g_by, d_by = {}, {}
    for a in gtd["annotations"]:
        g_by.setdefault(a["image_id"], []).append(a)
    for d in dets:
        d_by.setdefault(d["image_id"], []).append(d)
    scores, tps = [], []
    gt_rec = []  # (gt_ann, matched_score, matched_pred_cat)
    for im in gtd["images"]:
        G = g_by.get(im["id"], [])
        D = sorted(d_by.get(im["id"], []), key=lambda x: -x["score"])
        ious = _iou_matrix([g["bbox"] for g in G], [d["bbox"] for d in D])
        used = np.zeros(len(G), bool)
        gm = [(-1.0, None)] * len(G)
        for j, d in enumerate(D):
            best, bi = iou_thr, -1
            for i, g in enumerate(G):
                if used[i] or (class_aware and g["category_id"] != d["category_id"]):
                    continue
                if ious[i, j] >= best:
                    best, bi = ious[i, j], i
            scores.append(d["score"])
            if bi >= 0:
                used[bi] = True
                gm[bi] = (d["score"], d["category_id"])
                tps.append(1)
            else:
                tps.append(0)
        gt_rec += [(g, s, c) for g, (s, c) in zip(G, gm)]
    return np.array(scores), np.array(tps), gt_rec


def pr_at_best_f1(gt, dets, iou_thr=0.5):
    """Micro-averaged precision/recall/F1 at the confidence that maximises F1 (IoU 0.5)."""
    gtd = gt if isinstance(gt, dict) else load_json(gt)
    n_gt = len(gtd["annotations"])
    s, tp, _ = match_detections(gtd, dets, iou_thr)
    if len(s) == 0 or n_gt == 0:
        return dict(precision=0.0, recall=0.0, F1=0.0, conf_at_best_f1=None)
    o = np.argsort(-s)
    tp = tp[o]; s = s[o]
    ctp = np.cumsum(tp); cfp = np.cumsum(1 - tp)
    prec = ctp / (ctp + cfp); rec = ctp / n_gt
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    i = int(np.argmax(f1))
    return dict(precision=round(float(prec[i] * 100), 3), recall=round(float(rec[i] * 100), 3),
                F1=round(float(f1[i] * 100), 3), conf_at_best_f1=round(float(s[i]), 4))


def size_bucket_recall(gt, dets, conf, iou_thr=0.5):
    gtd = gt if isinstance(gt, dict) else load_json(gt)
    _, _, rec = match_detections(gtd, [d for d in dets if d["score"] >= conf], iou_thr)
    out = {}
    for k in ["tiny", "small", "medium", "large"]:
        lo, hi = AREA_RANGES[k]
        if k == "small":
            lo = 16 ** 2  # report 16-32 separately from tiny
        sel = [r for r in rec if lo <= r[0]["area"] < hi]
        out[k] = dict(n=len(sel), recall=round(100 * np.mean([r[1] >= 0 for r in sel]), 2) if sel else None)
    return out


def confusion_matrix(gt, dets, conf, iou_thr=0.5, class_names=CLASS_NAMES):
    """Rows = true class (+ 'background' row for FP), cols = predicted class (+ 'missed' column)."""
    gtd = gt if isinstance(gt, dict) else load_json(gt)
    D = [d for d in dets if d["score"] >= conf]
    _, _, rec = match_detections(gtd, D, iou_thr, class_aware=False)
    n = len(class_names)
    M = np.zeros((n + 1, n + 1), int)
    for g, s, c in rec:
        M[g["category_id"] - 1, (c - 1) if c is not None else n] += 1
    s, tp, _ = match_detections(gtd, D, iou_thr, class_aware=False)
    # unmatched predictions -> background row (by class)
    d_sorted = []
    by = {}
    for d in D:
        by.setdefault(d["image_id"], []).append(d)
    for im in gtd["images"]:
        d_sorted += sorted(by.get(im["id"], []), key=lambda x: -x["score"])
    for d, t in zip(d_sorted, tp):
        if not t:
            M[n, d["category_id"] - 1] += 1
    return M


# ----------------------------------------------------------------------------------
# Predictors: every family exposes predict(list_of_RGB_arrays, res) ->
#   list of (boxes_xyxy [N,4] in INPUT-image pixels, scores [N], labels0 [N])
# ----------------------------------------------------------------------------------
class UltralyticsPredictor:
    def __init__(self, weights, model_name="yolo11m", conf=0.001, iou=0.65, max_det=MAX_DETS, half=True):
        from ultralytics import YOLO, RTDETR
        self.model = RTDETR(weights) if model_name.startswith("rtdetr") else YOLO(weights)
        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.half = half and CFG.get("device") == "cuda"
        self.device = 0 if CFG.get("device") == "cuda" else "cpu"
        self.precision = "fp16" if self.half else "fp32"

    def predict(self, images, res):
        bgr = [np.ascontiguousarray(im[:, :, ::-1]) for im in images]
        # ultralytics 8.4: `half` is deprecated in favour of `quantize` (16 = fp16, None = fp32)
        rs = self.model.predict(bgr, imgsz=int(res), conf=self.conf, iou=self.iou, max_det=self.max_det,
                                quantize=16 if self.half else None, device=self.device, verbose=False)
        return [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)) for r in rs]

    def n_params(self):
        return sum(p.numel() for p in self.model.model.parameters())


def letterbox(im, new_long, stride=32, color=114):
    import cv2
    h, w = im.shape[:2]
    r = new_long / max(h, w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pw, ph = (math.ceil(nw / stride) * stride - nw), (math.ceil(nh / stride) * stride - nh)
    l, t = pw // 2, ph // 2
    im = cv2.copyMakeBorder(im, t, ph - t, l, pw - l, cv2.BORDER_CONSTANT, value=(color,) * 3)
    return im, r, (l, t)


class YOLOv7Predictor:
    """In-process inference with the official YOLOv7 repo (training runs as a subprocess)."""

    def __init__(self, weights, conf=0.001, iou=0.65, max_det=MAX_DETS):
        import torch
        repo = ensure_yolov7()
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import torch.nn as nn
        import models.yolo  # noqa: F401  (classes needed to unpickle the checkpoint)
        self.dev = torch.device(CFG.get("device", "cpu"))
        # (yolov7's attempt_load lower-cases paths, which breaks Google-Drive paths, so load directly)
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        self.model = (ck.get("ema") or ck["model"]).float().fuse().eval().to(self.dev)
        for mod in self.model.modules():
            if type(mod) is nn.Upsample:
                mod.recompute_scale_factor = None
            if type(mod).__name__ in ("Detect", "IDetect", "IAuxDetect", "IBin"):
                mod.inplace = True
        self.half = self.dev.type == "cuda"
        if self.half:
            self.model.half()
        self.stride = int(self.model.stride.max())
        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.precision = "fp16" if self.half else "fp32"

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def predict(self, images, res):
        import torch, torchvision
        out = []
        for im in images:
            lb, r, (l, t) = letterbox(im, int(res), self.stride)
            x = torch.from_numpy(lb.transpose(2, 0, 1).copy()).to(self.dev)
            x = (x.half() if self.half else x.float()) / 255.0
            with torch.no_grad():
                p = self.model(x[None])[0][0].float()  # [N, 5+nc]  xywh, obj, cls...
            sc = p[:, 5:] * p[:, 4:5]
            i, j = (sc > self.conf).nonzero(as_tuple=True)
            s = sc[i, j]
            if s.numel() > 30000:
                k = s.topk(30000).indices
                i, j, s = i[k], j[k], s[k]
            xywh = p[i, :4]
            b = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
            keep = torchvision.ops.batched_nms(b, s, j, self.iou)[: self.max_det]
            b, s, j = b[keep], s[keep], j[keep]
            b[:, [0, 2]] -= l; b[:, [1, 3]] -= t; b /= r
            b[:, [0, 2]] = b[:, [0, 2]].clamp(0, im.shape[1]); b[:, [1, 3]] = b[:, [1, 3]].clamp(0, im.shape[0])
            out.append((b.cpu().numpy(), s.cpu().numpy(), j.cpu().numpy().astype(int)))
        return out


class TorchvisionPredictor:
    def __init__(self, model):
        import torch
        self.model = model.eval().to(CFG.get("device", "cpu"))
        self.model.roi_heads.score_thresh = 0.001
        self.model.roi_heads.detections_per_img = MAX_DETS
        self.amp = CFG.get("device") == "cuda"
        self.precision = "fp16-autocast" if self.amp else "fp32"

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def predict(self, images, res):
        import torch
        set_frcnn_resolution(self.model, res, images[0].shape[1], images[0].shape[0])
        dev = CFG.get("device", "cpu")
        xs = [torch.from_numpy(im.transpose(2, 0, 1).copy()).to(dev).float() / 255.0 for im in images]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
            outs = self.model(xs)
        return [(o["boxes"].float().cpu().numpy(), o["scores"].float().cpu().numpy(), o["labels"].cpu().numpy().astype(int) - 1)
                for o in outs]


class HFDetrPredictor:
    def __init__(self, model, model_name):
        self.model = model.eval().to(CFG.get("device", "cpu"))
        self.sigmoid = model_name == "deformable_detr"
        self.amp = CFG.get("device") == "cuda"
        self.precision = "fp16-autocast" if self.amp else "fp32"

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def predict(self, images, res):
        import torch
        dev = CFG.get("device", "cpu")
        aspect = images[0].shape[0] / images[0].shape[1]  # one tensor size per batch; boxes are normalised per image
        x = torch.stack([hf_preprocess(im, res, aspect) for im in images]).to(dev)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
            o = self.model(pixel_values=x, pixel_mask=torch.ones(x.shape[0], x.shape[2], x.shape[3], dtype=torch.long, device=dev))
        logits, boxes = o.logits.float(), o.pred_boxes.float()
        out = []
        for b in range(len(images)):
            prob = logits[b].sigmoid() if self.sigmoid else logits[b].softmax(-1)[:, :-1]
            Q, C = prob.shape
            k = min(MAX_DETS, Q * C)
            s, idx = prob.flatten().topk(k)
            q, c = idx // C, idx % C
            cx, cy, w, h = boxes[b, q].unbind(-1)
            H, W = images[b].shape[:2]
            xyxy = torch.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], -1)
            out.append((xyxy.cpu().numpy(), s.cpu().numpy(), c.cpu().numpy().astype(int)))
        return out


# ----------------------------------------------------------------------------------
# Dataset-level prediction (full frame or sliced) -> COCO detections in NATIVE coords
# ----------------------------------------------------------------------------------
def _to_coco(img_id, boxes, scores, labels0, sx=1.0, sy=1.0, min_score=0.001):
    out = []
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, labels0):
        if s < min_score:
            continue
        out.append(dict(image_id=int(img_id), category_id=int(c) + 1,
                        bbox=[round(float(x1 * sx), 2), round(float(y1 * sy), 2),
                              round(float((x2 - x1) * sx), 2), round(float((y2 - y1) * sy), 2)],
                        score=round(float(s), 5)))
    return out


def merge_nms(boxes, scores, labels, iou=0.5, max_det=MAX_DETS):
    import torch, torchvision
    if len(boxes) == 0:
        return boxes, scores, labels
    b = torch.as_tensor(np.asarray(boxes), dtype=torch.float32)
    s = torch.as_tensor(np.asarray(scores), dtype=torch.float32)
    l = torch.as_tensor(np.asarray(labels), dtype=torch.int64)
    keep = torchvision.ops.batched_nms(b, s, l, iou)[:max_det].numpy()
    return np.asarray(boxes)[keep], np.asarray(scores)[keep], np.asarray(labels)[keep]


def merge_fragments(boxes, scores, labels, tile_ids, frag, ios_thr=0.5):
    """Objects cut by an interior tile border come back as a partial box from one tile and a (partial or complete)
    box from the neighbouring tile. Their IoU is often < 0.5, so NMS keeps both and the fragment becomes a false
    positive (a *perfect* detector lost ~8 mAP50-95 / APS on the synthetic set). SAHI-style greedy merge on
    intersection-over-smaller (IoS), restricted to (a) boxes flagged as touching an interior tile edge, (b) pairs from
    different tiles of the same class: fragment -> absorbed by the complete box, fragment + fragment -> union."""
    B = np.asarray(boxes, float).copy(); S = np.asarray(scores, float).copy(); L = np.asarray(labels)
    T = np.asarray(tile_ids); F = np.asarray(frag, bool).copy()
    alive = np.ones(len(B), bool)
    area = lambda b: np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    for i in np.argsort(area(B)):  # smallest fragments first
        if not (F[i] and alive[i]):
            continue
        cand = np.flatnonzero(alive & (L == L[i]) & (T != T[i]))
        if not len(cand):
            continue
        bj = B[cand]
        iw = np.clip(np.minimum(B[i, 2], bj[:, 2]) - np.maximum(B[i, 0], bj[:, 0]), 0, None)
        ih = np.clip(np.minimum(B[i, 3], bj[:, 3]) - np.maximum(B[i, 1], bj[:, 1]), 0, None)
        ios = iw * ih / np.maximum(np.minimum(area(B[i]), area(bj)), 1e-9)
        k = int(np.argmax(ios))
        if ios[k] < ios_thr:
            continue
        j = cand[k]
        if F[j]:  # both partial: the object spans the tile border -> union box
            B[j] = [min(B[i, 0], B[j, 0]), min(B[i, 1], B[j, 1]), max(B[i, 2], B[j, 2]), max(B[i, 3], B[j, 3])]
        S[j] = max(S[i], S[j])
        alive[i] = False
    return B[alive], S[alive], L[alive]


def sliced_predict(predictor, img, tile=640, overlap=0.2, tile_res=None, full_res=None, batch=8, merge_iou=0.5,
                   edge_px=2.0):
    """SAHI-style sliced inference on one native RGB image. If full_res is given the same model is
    also run on the whole frame at that size and merged (the 'sliced+full' variant).
    Merging: merge_fragments() (objects cut by a tile border), then class-wise NMS (duplicates seen whole in two
    overlapping tiles), capped at MAX_DETS."""
    H, W = img.shape[:2]
    grid = tile_grid(W, H, tile, overlap)
    B, S, L, T, F = [], [], [], [], []
    for i in range(0, len(grid), batch):
        chunk = grid[i:i + batch]
        crops = [img[y:y + h, x:x + w] for (x, y, w, h) in chunk]
        for t, ((x, y, w, h), (b, s, l)) in enumerate(zip(chunk, predictor.predict(crops, tile_res or tile))):
            if len(b):
                b = np.asarray(b, float)
                # touches a tile edge that is NOT an image border -> possibly a cut object
                fr = (((b[:, 0] <= edge_px) & (x > 0)) | ((b[:, 2] >= w - edge_px) & (x + w < W)) |
                      ((b[:, 1] <= edge_px) & (y > 0)) | ((b[:, 3] >= h - edge_px) & (y + h < H)))
                b = b.copy(); b[:, [0, 2]] += x; b[:, [1, 3]] += y
                B.append(b); S.append(s); L.append(l); T.append(np.full(len(b), i + t)); F.append(fr)
    if full_res:
        b, s, l = predictor.predict([img], full_res)[0]
        B.append(b); S.append(s); L.append(l); T.append(np.full(len(b), -1)); F.append(np.zeros(len(b), bool))
    if not B:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    B, S, L, T, F = map(np.concatenate, (B, S, L, T, F))
    if not len(B):
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    # fragments first (otherwise NMS may keep a confident fragment and suppress the complete box), then NMS
    B, S, L = merge_fragments(B, S, L, T, F)
    return merge_nms(B, S, L, merge_iou)


def predict_dataset(predictor, gt_name, pack, res, mode="full", tile=640, overlap=0.2, full_res=None, batch=4):
    """Runs the predictor over all images of a canonical dataset.
    mode='full'   : whole frame from `pack` resized to `res` (boxes rescaled to native coords)
    mode='sliced' : tiles from the native-resolution pack ('aerial_native'), optional full_res merge."""
    d = load_json(json_path(gt_name))
    ensure_pack(pack)
    dets, t0 = [], time.time()
    ims = d["images"]
    if mode == "full":
        for i in range(0, len(ims), batch):
            chunk = ims[i:i + batch]
            arrs = [read_rgb(image_path(pack, im["file_name"])) for im in chunk]
            for im, arr, (b, s, l) in zip(chunk, arrs, predictor.predict(arrs, res)):
                dets += _to_coco(im["id"], b, s, l, im["width"] / arr.shape[1], im["height"] / arr.shape[0])
    else:
        for im in ims:
            arr = read_rgb(image_path(pack, im["file_name"]))
            b, s, l = sliced_predict(predictor, arr, tile, overlap, tile_res=res, full_res=full_res)
            dets += _to_coco(im["id"], b, s, l, im["width"] / arr.shape[1], im["height"] / arr.shape[0])
    print(f"  predicted {len(ims)} images ({mode}) in {time.time() - t0:.0f}s, {len(dets)} detections")
    return dets


def benchmark_predictor(predictor, gt_name, pack, res, n=30, warmup=5, mode="full", tile=640, overlap=0.2, full_res=None):
    """Batch-1 latency (ms/img, preprocessing+model+NMS, excluding disk IO) and peak inference memory."""
    import torch
    d = load_json(json_path(gt_name))
    arrs = [read_rgb(image_path(pack, im["file_name"])) for im in d["images"][:max(3, min(10, len(d['images'])))]]
    cuda = torch.cuda.is_available()
    if cuda:
        free_gpu()
    run = (lambda a: predictor.predict([a], res)) if mode == "full" else \
        (lambda a: sliced_predict(predictor, a, tile, overlap, tile_res=res, full_res=full_res))
    n = 5 if CFG.get("smoke") else n
    for i in range(warmup if not CFG.get("smoke") else 1):
        run(arrs[i % len(arrs)])
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n):
        run(arrs[i % len(arrs)])
    if cuda:
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000
    mem = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if cuda else None
    return dict(latency_ms=round(ms, 2), FPS=round(1000 / ms, 2), GPU_memory_infer_GB=mem)

# ----------------------------------------------------------------------------------
# Generic COCO dataset for torchvision / HF trainers
# ----------------------------------------------------------------------------------
def _res_hw(res, aspect):
    """(H, W) for a long side `res` and aspect = H/W, rounded to multiples of 32."""
    W = int(res)
    H = max(32, int(round(res * aspect / 32)) * 32)
    return H, W


IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def hf_preprocess(im, res, aspect=None):
    import torch, cv2
    h, w = im.shape[:2]
    H, W = _res_hw(res, aspect if aspect else h / w)
    x = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = (x - IMNET_MEAN) / IMNET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class CocoDetDataset:
    """Reads canonical COCO + pack images. Returns RGB uint8 array and native-scaled targets
    rescaled to the stored image size. Horizontal flip augmentation when train=True."""

    def __init__(self, name, pack, train=True, seed=SEED):
        self.d = load_json(json_path(name))
        self.pack, self.train = pack, train
        self.by = {}
        for a in self.d["annotations"]:
            self.by.setdefault(a["image_id"], []).append(a)
        self.rng = np.random.RandomState(seed)
        # resolve paths here: DataLoader workers started with 'spawn'/'forkserver' re-import this module with an
        # empty CFG, so __getitem__ must not depend on global state
        self.paths = [image_path(pack, im["file_name"]) for im in self.d["images"]]

    def __len__(self):
        return len(self.d["images"])

    def __getitem__(self, i):
        im = self.d["images"][i]
        arr = read_rgb(self.paths[i])
        sx, sy = arr.shape[1] / im["width"], arr.shape[0] / im["height"]
        A = self.by.get(im["id"], [])
        boxes = np.array([[a["bbox"][0] * sx, a["bbox"][1] * sy, (a["bbox"][0] + a["bbox"][2]) * sx,
                           (a["bbox"][1] + a["bbox"][3]) * sy] for a in A], np.float32).reshape(-1, 4)
        labels = np.array([a["category_id"] for a in A], np.int64)
        if self.train and self.rng.rand() < 0.5:
            arr = np.ascontiguousarray(arr[:, ::-1])
            W = arr.shape[1]
            boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        return arr, boxes, labels, im["id"]


def _collate(b):
    return list(zip(*b))


# ----------------------------------------------------------------------------------
# Model builders
# ----------------------------------------------------------------------------------
def set_frcnn_resolution(model, res, w, h):
    """Resolution = longest side. torchvision resizes so that min side == min_size unless the
    long side would exceed max_size, so set both from the image aspect."""
    model.transform.max_size = int(res)
    model.transform.min_size = (int(round(res * min(w, h) / max(w, h))),)


def build_frcnn(name, num_classes=NUM_CLASSES, small_anchors=True, pretrained=True):
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2, FasterRCNN
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    kw = dict(box_detections_per_img=MAX_DETS)
    if name == "frcnn_r50_v1":
        m = fasterrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None, **kw)
    elif name == "frcnn_r50_v2":
        m = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None, **kw)
    elif name == "frcnn_r101":
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        from torchvision.models import ResNet101_Weights
        bb = resnet_fpn_backbone(backbone_name="resnet101", weights=ResNet101_Weights.IMAGENET1K_V2 if pretrained else None,
                                 trainable_layers=3)
        m = FasterRCNN(bb, num_classes=91, **kw)
        if pretrained:  # transplant COCO neck/RPN/ROI heads (identical shapes for R50/R101 FPN)
            donor = fasterrcnn_resnet50_fpn(weights="DEFAULT").state_dict()
            sd = {k: v for k, v in donor.items() if not k.startswith("backbone.body.")}
            missing, unexpected = m.load_state_dict(sd, strict=False)
            assert not unexpected, unexpected
    else:
        raise ValueError(name)
    if small_anchors:  # aerial objects: shift anchors one octave down (same #anchors -> RPN weights still fit)
        m.rpn.anchor_generator = AnchorGenerator(sizes=((16,), (32,), (64,), (128,), (256,)),
                                                 aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    inf = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(inf, num_classes + 1)
    return m


def build_hf(name, num_classes=NUM_CLASSES, class_names=CLASS_NAMES):
    from transformers import DetrForObjectDetection, DeformableDetrForObjectDetection
    import transformers, logging as _lg
    id2label = {i: n for i, n in enumerate(class_names)}
    label2id = {n: i for i, n in id2label.items()}
    transformers.logging.set_verbosity_error()
    if name == "detr_r50":
        m = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", num_labels=num_classes, id2label=id2label,
                                                   label2id=label2id, num_queries=MAX_DETS, ignore_mismatched_sizes=True)
    elif name == "deformable_detr":
        m = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr", num_labels=num_classes,
                                                             id2label=id2label, label2id=label2id,
                                                             ignore_mismatched_sizes=True)
    else:
        raise ValueError(name)
    return m


def load_matching_state_dict(model, sd):
    own = model.state_dict()
    ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(ok, strict=False)
    print(f"  loaded {len(ok)}/{len(own)} tensors from checkpoint (shape-matched)")
    return model


# ----------------------------------------------------------------------------------
# Trainers. All write into run_dir (on Drive) and are resumable.
# Each returns dict(weights=..., train_time_h=..., GPU_memory_train_GB=..., batch_size=...)
# ----------------------------------------------------------------------------------
class _Timer:
    def __init__(self, run_dir):
        self.p = os.path.join(run_dir, "train_time.json")
        self.prev = load_json(self.p)["seconds"] if os.path.exists(self.p) else 0.0
        self.t0 = time.time()

    def save(self):
        save_json({"seconds": self.prev + time.time() - self.t0}, self.p)

    @property
    def hours(self):
        return round((self.prev + time.time() - self.t0) / 3600, 3)


def _is_oom(e):
    return "out of memory" in str(e).lower() or e.__class__.__name__ == "OutOfMemoryError"


def resolve_init(init, model_name):
    """'coco' -> None (use the family's default pretrained weights); 'exp:<id>' -> weights path."""
    if init in (None, "coco"):
        return None
    if init.startswith("exp:"):
        rec = load_record(init[4:])
        if rec is None or not rec.get("checkpoint_path"):
            raise FileNotFoundError(f"init experiment {init} has no checkpoint yet")
        if rec["model_version"] != model_name:
            print(f"  NOTE: initialising {model_name} from a {rec['model_version']} checkpoint (shape-matched)")
        return rec["checkpoint_path"]
    return init


def train_ultralytics(spec, run_dir):
    import torch
    from ultralytics import YOLO, RTDETR
    m = spec["model_version"]
    yml = make_yolo_dataset(spec["data_key"], spec["train_set"], spec["val_set"], spec["train_pack"], spec["class_names"],
                            val_pack=spec["eval_pack"])
    wdir = os.path.join(run_dir, "train", "weights")
    last = os.path.join(wdir, "last.pt")
    done = os.path.join(run_dir, "TRAIN_DONE")
    timer = _Timer(run_dir)
    os.chdir(os.path.join(CFG["local_root"], "weights"))  # pretrained .pt files are downloaded here
    cls = RTDETR if m.startswith("rtdetr") else YOLO
    batch = spec["batch_size"]
    with GPUMemPoller() as poll:
        while not os.path.exists(done):
            try:
                if os.path.exists(last):
                    print("Resuming from", last)
                    cls(last).train(resume=True)
                else:
                    init = resolve_init(spec.get("init"), m) or f"{m}.pt"
                    E = spec["epochs"]
                    cls(init).train(
                        data=yml, imgsz=train_res(spec), epochs=E, batch=batch, project=run_dir, name="train",
                        exist_ok=True, seed=spec["seed"], deterministic=True, workers=2, cache=False,
                        amp=CFG["device"] == "cuda", device=0 if CFG["device"] == "cuda" else "cpu",
                        val=False, plots=False, close_mosaic=min(10, max(0, E // 4)), pretrained=True, verbose=False)
                Path(done).write_text(now())
            except Exception as e:
                if _is_oom(e) and batch > 1 and not os.path.exists(last):
                    batch = max(1, batch // 2)
                    print(f"CUDA OOM -> retrying with batch {batch}")
                    shutil.rmtree(os.path.join(run_dir, "train"), ignore_errors=True)
                    free_gpu()
                    continue
                timer.save()
                raise
    timer.save()
    w = last if os.path.exists(last) else os.path.join(wdir, "best.pt")
    return dict(weights=w, train_time_h=timer.hours, GPU_memory_train_GB=poll.peak_gb, batch_size=batch)


# ---- YOLOv7 (official repo, run as a subprocess) ----------------------------------
YOLOV7_COMMIT = "a207844b1ce82d204ab36d87d496728d3d2348e7"


def ensure_yolov7():
    repo = os.path.join(CFG["local_root"], "yolov7")
    if not os.path.exists(os.path.join(repo, "train.py")):
        subprocess.run(["git", "clone", "-q", "https://github.com/WongKinYiu/yolov7.git", repo], check=True)
        subprocess.run(["git", "-C", repo, "checkout", "-q", YOLOV7_COMMIT], check=False)
    _patch_yolov7(repo)
    w = os.path.join(repo, "yolov7_training.pt")
    if not os.path.exists(w):
        import urllib.request
        urllib.request.urlretrieve("https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt", w + ".part")
        os.replace(w + ".part", w)
    return repo


def _patch_yolov7(repo):
    """Idempotent compatibility patches for modern torch / numpy (no behaviour change)."""
    patches = {
        "utils/loss.py": [("gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)",
                           "gj.clamp_(0, gain[3].long() - 1), gi.clamp_(0, gain[2].long() - 1)"),
                          ("from_which_layer.append(torch.ones(size=(len(b),)) * i)",
                           "from_which_layer.append((torch.ones(size=(len(b),)) * i).to(targets.device))")],
        "utils/datasets.py": [(".astype(np.int)", ".astype(int)")],
        "utils/general.py": [("np.int)", "int)")],
        "utils/plots.py": [("np.int)", "int)")],
        "utils/metrics.py": [("np.trapz(", "(getattr(np, 'trapezoid', None) or np.trapz)(")],
    }
    for f, reps in patches.items():
        p = os.path.join(repo, f)
        if not os.path.exists(p):
            continue
        s = Path(p).read_text()
        s2 = s
        for a, b in reps:
            s2 = s2.replace(a, b)
        if s2 != s:
            Path(p).write_text(s2)


def train_yolov7(spec, run_dir):
    repo = ensure_yolov7()
    # separate folder from Ultralytics: both write labels/*.cache there, in incompatible formats
    # (YOLOv7 torch.load()s an Ultralytics cache -> UnpicklingError)
    key = spec["data_key"] + "__v7"
    make_yolo_dataset(key, spec["train_set"], spec["val_set"], spec["train_pack"], spec["class_names"],
                      val_pack=spec["eval_pack"])
    yml = os.path.join(CFG["local_root"], "yolo", key + ("_smoke" if CFG.get("smoke") else ""), "data_yolov7.yaml")
    last = os.path.join(run_dir, "train", "weights", "last.pt")
    done = os.path.join(run_dir, "TRAIN_DONE")
    init = resolve_init(spec.get("init"), "yolov7") or os.path.join(repo, "yolov7_training.pt")
    timer = _Timer(run_dir)
    batch = spec["batch_size"]
    env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", WANDB_MODE="disabled", PYTHONWARNINGS="ignore")
    dev = "0" if CFG["device"] == "cuda" else "cpu"
    with GPUMemPoller() as poll:
        while not os.path.exists(done):
            if os.path.exists(last):
                cmd = [sys.executable, "train.py", "--resume", last]
            else:
                cmd = [sys.executable, "train.py", "--workers", "2", "--device", dev, "--batch-size", str(batch),
                       "--data", yml, "--img", str(train_res(spec)), str(train_res(spec)),
                       "--cfg", "cfg/training/yolov7.yaml", "--weights", init, "--hyp", "data/hyp.scratch.p5.yaml",
                       "--project", run_dir, "--name", "train", "--exist-ok", "--epochs", str(spec["epochs"]), "--notest"]
            print(" ".join(cmd))
            p = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)
            log = p.stdout + "\n" + p.stderr
            Path(run_dir, f"yolov7_log_{int(time.time())}.txt").write_text(log[-200000:])
            if p.returncode == 0 and os.path.exists(last):
                Path(done).write_text(now())
                for extra in glob.glob(os.path.join(run_dir, "train", "weights", "epoch_*.pt")) + \
                        glob.glob(os.path.join(run_dir, "train", "weights", "init.pt")):
                    os.remove(extra)  # YOLOv7 keeps several full checkpoints; free Drive space
                break
            if "out of memory" in log.lower() and batch > 1 and not os.path.exists(last):
                batch //= 2
                print(f"CUDA OOM -> retrying with batch {batch}")
                shutil.rmtree(os.path.join(run_dir, "train"), ignore_errors=True)
                continue
            timer.save()
            print(log[-4000:])
            raise RuntimeError("YOLOv7 training failed — see the log printed above / in run_dir")
    timer.save()
    return dict(weights=last, train_time_h=timer.hours, GPU_memory_train_GB=poll.peak_gb, batch_size=batch)


# ---- torchvision Faster R-CNN and HF DETR: shared PyTorch loop -------------------
def _torch_loop(spec, run_dir, model, step_fn, param_groups, epochs, batch, amp, clip, lr_sched="cosine",
                warmup_iters=300, optim="sgd"):
    import torch
    from torch.utils.data import DataLoader
    dev = CFG["device"]
    model.to(dev)
    ds = CocoDetDataset(spec["train_set"], spec["train_pack"], train=True, seed=spec["seed"])
    g = torch.Generator()
    g.manual_seed(spec["seed"])
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2, collate_fn=_collate, generator=g,
                    drop_last=len(ds) > batch, persistent_workers=False)
    if optim == "sgd":
        opt = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    total = epochs * len(dl)
    base = [pg["lr"] for pg in opt.param_groups]

    def lr_at(it):
        w = min(1.0, (it + 1) / max(1, min(warmup_iters, total // 5)))
        if lr_sched == "cosine":
            f = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * it / max(1, total)))
        else:  # step at 2/3
            f = 1.0 if it < int(total * 2 / 3) else 0.1
        return w * f

    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ck = os.path.join(run_dir, "ckpt_last.pt")
    final = os.path.join(run_dir, "final.pt")
    if os.path.exists(os.path.join(run_dir, "TRAIN_DONE")) and os.path.exists(final):
        model.load_state_dict(torch.load(final, map_location="cpu", weights_only=False))
        print("Training already finished ->", final)
        return final, _Timer(run_dir).hours
    start = 0
    if os.path.exists(ck):
        s = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); scaler.load_state_dict(s["scaler"])
        start = s["epoch"] + 1
        print(f"Resumed from epoch {start}")
    log = os.path.join(run_dir, "train_log.csv")
    timer = _Timer(run_dir)
    it = start * len(dl)
    for ep in range(start, epochs):
        model.train()
        t0, losses = time.time(), []
        for imgs, boxes, labels, _ in dl:
            for pg, b in zip(opt.param_groups, base):
                pg["lr"] = b * lr_at(it)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loss = step_fn(model, imgs, boxes, labels)
            if not torch.isfinite(loss):
                print("  non-finite loss, batch skipped")
                opt.zero_grad(set_to_none=True)
                it += 1
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss))
            it += 1
        msg = f"epoch {ep + 1}/{epochs} loss {np.mean(losses) if losses else float('nan'):.4f} ({time.time() - t0:.0f}s)"
        print(msg)
        with open(log, "a") as f:
            f.write(f"{ep + 1},{np.mean(losses) if losses else ''},{time.time() - t0:.1f}\n")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), epoch=ep), ck + ".tmp")
        os.replace(ck + ".tmp", ck)
        timer.save()
    torch.save(model.state_dict(), final)
    Path(run_dir, "TRAIN_DONE").write_text(now())
    if os.path.exists(ck):
        os.remove(ck)  # optimizer state no longer needed; saves ~2x model size on Drive
    return final, timer.hours


def train_torchvision(spec, run_dir):
    import torch
    model = build_frcnn(spec["model_version"], spec["num_classes"], spec.get("small_anchors", True))
    init = resolve_init(spec.get("init"), spec["model_version"])
    if init:
        load_matching_state_dict(model, torch.load(init, map_location="cpu", weights_only=False))
    ds_aspect = None

    def step(m, imgs, boxes, labels):
        nonlocal ds_aspect
        dev = CFG["device"]
        h, w = imgs[0].shape[:2]
        set_frcnn_resolution(m, train_res(spec), w, h)
        x = [torch.from_numpy(i.transpose(2, 0, 1).copy()).to(dev).float() / 255.0 for i in imgs]
        t = [dict(boxes=torch.from_numpy(b).to(dev), labels=torch.from_numpy(l).to(dev)) for b, l in zip(boxes, labels)]
        ld = m(x, t)
        return sum(ld.values())

    batch = spec["batch_size"]
    lr = 0.02 * batch / 16
    with GPUMemPoller() as poll:
        while True:
            try:
                params = [p for p in model.parameters() if p.requires_grad]
                final, hours = _torch_loop(spec, run_dir, model, step, [dict(params=params, lr=lr)], spec["epochs"], batch,
                                           amp=CFG["device"] == "cuda", clip=10.0, optim="sgd")
                break
            except Exception as e:
                if _is_oom(e) and batch > 1 and not os.path.exists(os.path.join(run_dir, "ckpt_last.pt")):
                    batch //= 2; lr = 0.02 * batch / 16
                    print(f"CUDA OOM -> retrying with batch {batch}")
                    free_gpu(); continue
                raise
    return dict(weights=final, train_time_h=hours, GPU_memory_train_GB=poll.peak_gb, batch_size=batch)


def train_hf(spec, run_dir):
    import torch
    model = build_hf(spec["model_version"], spec["num_classes"], spec["class_names"])
    init = resolve_init(spec.get("init"), spec["model_version"])
    if init:
        load_matching_state_dict(model, torch.load(init, map_location="cpu", weights_only=False))
    res = train_res(spec)

    def step(m, imgs, boxes, labels):
        dev = CFG["device"]
        aspect = imgs[0].shape[0] / imgs[0].shape[1]  # mixed-size batches (e.g. VisDrone) share one tensor size
        x = torch.stack([hf_preprocess(i, res, aspect) for i in imgs]).to(dev)
        tg = []
        for i, b, l in zip(imgs, boxes, labels):
            H, W = i.shape[:2]
            bb = torch.from_numpy(b).float()
            cxcywh = torch.stack([(bb[:, 0] + bb[:, 2]) / 2 / W, (bb[:, 1] + bb[:, 3]) / 2 / H,
                                  (bb[:, 2] - bb[:, 0]) / W, (bb[:, 3] - bb[:, 1]) / H], -1) if len(bb) else torch.zeros(0, 4)
            tg.append(dict(class_labels=torch.from_numpy(l - 1).long().to(dev), boxes=cxcywh.clamp(0, 1).to(dev)))
        out = m(pixel_values=x, pixel_mask=torch.ones(x.shape[0], x.shape[2], x.shape[3], dtype=torch.long, device=dev), labels=tg)
        return out.loss

    bb = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    batch = spec["batch_size"]
    amp = spec.get("amp", False) and CFG["device"] == "cuda"  # fp32 by default: DETR matching is fragile in fp16
    with GPUMemPoller() as poll:
        while True:
            try:
                final, hours = _torch_loop(spec, run_dir, model, step, [dict(params=rest, lr=1e-4), dict(params=bb, lr=1e-5)],
                                           spec["epochs"], batch, amp=amp, clip=0.1, lr_sched="step", optim="adamw",
                                           warmup_iters=0)
                break
            except Exception as e:
                if _is_oom(e) and batch > 1 and not os.path.exists(os.path.join(run_dir, "ckpt_last.pt")):
                    batch //= 2
                    print(f"CUDA OOM -> retrying with batch {batch}")
                    free_gpu(); continue
                raise
    return dict(weights=final, train_time_h=hours, GPU_memory_train_GB=poll.peak_gb, batch_size=batch)


def load_predictor(spec, weights):
    import torch
    fam = MODEL_FAMILY[spec["model_version"]]
    if fam == "ultralytics":
        return UltralyticsPredictor(weights, spec["model_version"])
    if fam == "yolov7":
        return YOLOv7Predictor(weights)
    if fam == "torchvision":
        m = build_frcnn(spec["model_version"], spec["num_classes"], spec.get("small_anchors", True), pretrained=False)
        m.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False))
        return TorchvisionPredictor(m)
    if fam == "hf":
        m = build_hf(spec["model_version"], spec["num_classes"], spec["class_names"])
        m.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False))
        return HFDetrPredictor(m, spec["model_version"])
    raise ValueError(fam)


TRAINERS = {"ultralytics": train_ultralytics, "yolov7": train_yolov7, "torchvision": train_torchvision, "hf": train_hf}

# ----------------------------------------------------------------------------------
# Experiment specs, IDs, records
# ----------------------------------------------------------------------------------
def _short(train_set):
    m = re.match(r"aerial_train_(\d+)$", train_set)
    if m:
        return f"n{m.group(1)}"
    m = re.match(r"ground_train_(\d+)$", train_set)
    if m:
        return f"g{m.group(1)}"
    # derived datasets are passed as absolute paths: use the file stem, otherwise every path truncates
    # to the same 40-character Drive prefix and the ids collide
    name = Path(train_set).stem if (os.path.isabs(train_set) or train_set.endswith(".json")) else train_set.replace("ext/", "")
    return re.sub(r"[^A-Za-z0-9]+", "-", name)[:40]


def train_res(spec):
    """Input size the network is trained at (== spec['resolution'] except in smoke mode)."""
    return int(spec.get("train_res") or spec["resolution"])


def make_spec(model, train_set="aerial_train_1000", resolution=960, stage="S1", epochs=None, batch_size=None,
              seed=SEED, init="coco", pretraining="COCO", tiling="none", tile=640, overlap=0.2, full_res=None,
              val_set="aerial_val", test_set="aerial_test", train_pack="aerial_1280", eval_pack="aerial_1280",
              native_pack="aerial_native", class_names=CLASS_NAMES, dataset_size=None, notes="", tag="",
              small_anchors=True, amp=False):
    """Create an experiment specification. tiling: 'none' | 'tile_train' (train on native tiles,
    evaluate with sliced inference)."""
    assert model in MODEL_FAMILY, f"unknown model {model}; choose from {list(MODEL_FAMILY)}"
    res = int(resolution)
    if tiling == "tile_train":
        res = tile
    if dataset_size is None:
        m = re.search(r"_(\d+)$", train_set)
        dataset_size = int(m.group(1)) if m else None
    E = epochs or DEFAULT_EPOCHS[model]
    B = batch_size or T4_BATCH.get(model, {}).get(min(T4_BATCH.get(model, {640: 4}), key=lambda r: abs(r - res)), 4)
    # `resolution` is the NOMINAL input size (used in the experiment id, records and notebook filters);
    # `train_res` is what the network actually sees. They differ only in smoke mode with MAVREC_SMOKE_MAX_RES,
    # so smoke runs exercise exactly the same ids / filters / decisions as the real runs.
    train_res = res
    if CFG.get("smoke"):
        E = 1
        if CFG.get("smoke_max_res"):
            train_res = min(res, CFG["smoke_max_res"]); tile = min(tile, CFG["smoke_max_res"])
            if tiling == "tile_train":
                train_res = tile
        B = min(B, 2)
    spec = dict(stage=stage, model_version=model, model_family=MODEL_GROUP[model], framework=MODEL_FAMILY[model],
                backbone=MODEL_BACKBONE[model], train_set=train_set, val_set=val_set, test_set=test_set,
                train_pack=train_pack, eval_pack=eval_pack, native_pack=native_pack, dataset_size=dataset_size,
                resolution=res, train_res=train_res, epochs=E, batch_size=B, seed=seed, init=init, pretraining=pretraining,
                tiling=tiling, tile=tile, overlap=overlap, full_res=full_res, class_names=list(class_names),
                num_classes=len(class_names), notes=notes, small_anchors=small_anchors, amp=amp,
                pretrained_source=PRETRAINED_SOURCE[model])
    pre = re.sub(r"[^A-Za-z0-9]+", "", pretraining.replace(">", "2"))
    spec["experiment_id"] = (f"{stage}_{model}_{_short(train_set)}_r{res}_{'tile' if tiling == 'tile_train' else 'full'}"
                             f"_{pre}_s{seed}" + (f"_{tag}" if tag else ""))
    return spec


def record_path(exp_id):
    return os.path.join(CFG["records_dir"], f"{exp_id}.json")


def load_record(exp_id):
    p = record_path(exp_id)
    return load_json(p) if os.path.exists(p) else None


def save_record(rec):
    rec["updated"] = now()
    save_json(rec, record_path(rec["experiment_id"]), indent=1)
    return rec


def run_dir_for(exp_id):
    d = os.path.join(CFG["runs_dir"], exp_id)
    os.makedirs(d, exist_ok=True)
    return d


def _prepare_training_data(spec):
    """Resolves tile datasets and the YOLO conversion key."""
    if spec["tiling"] == "tile_train" and not spec["train_set"].startswith("tiles_"):
        jpath, pack = build_tile_dataset(spec["train_set"], spec["tile"], spec["overlap"], native_pack=spec["native_pack"],
                                         seed=spec["seed"])
        spec["parent_train_set"] = spec["train_set"]
        spec["train_set"], spec["train_pack"] = jpath, pack
    else:
        ensure_pack(spec["train_pack"])
    spec["data_key"] = re.sub(r"[^A-Za-z0-9_]+", "-", f"{Path(spec['train_set']).stem}__{Path(spec['val_set']).stem}__"
                                                     f"{spec['train_pack']}" + (f"__v-{spec['eval_pack']}" if spec['eval_pack'] != spec['train_pack'] else ""))
    return spec


def train(spec, force=False):
    """Trains (or resumes) an experiment. Returns the train-info dict."""
    exp_id = spec["experiment_id"]
    rd = run_dir_for(exp_id)
    info_p = os.path.join(rd, "train_info.json")
    if os.path.exists(info_p) and not force:
        info = load_json(info_p)
        if os.path.exists(info["weights"]):
            print(f"[{exp_id}] already trained -> {info['weights']}")
            return info
    done = os.path.join(rd, "TRAIN_DONE")
    if os.path.exists(done):
        w = load_json(info_p)["weights"] if os.path.exists(info_p) else None
        if not (w and os.path.exists(w)):
            # seen once on Colab: a run marked done whose checkpoint later vanished from Drive. Without this, the
            # trainers skip training (TRAIN_DONE) and hand back a path that does not exist.
            print(f"[{exp_id}] WARNING: marked trained but checkpoint {w} is missing on Drive -> retraining")
            for f in (done, info_p):
                if os.path.exists(f):
                    os.remove(f)
    spec = _prepare_training_data(dict(spec))
    save_json(spec, os.path.join(rd, "spec.json"), indent=1)
    print(f"[{exp_id}] training {spec['model_version']} | data={Path(spec['train_set']).stem} | res={spec['resolution']} "
          f"| epochs={spec['epochs']} | batch={spec['batch_size']} | init={spec['init']}"
          + (f" | smoke input size {train_res(spec)}" if train_res(spec) != spec["resolution"] else ""))
    set_seed(spec["seed"])
    free_gpu()
    info = TRAINERS[spec["framework"]](spec, rd)
    info["train_set_resolved"] = spec["train_set"]
    save_json(info, info_p, indent=1)
    free_gpu()
    return info


def evaluate(exp_id_or_spec, split="val", mode=None, res=None, full_res=None, save_preds=True, force=False,
             predictor=None, gt_name=None, eval_pack=None, cat_ids=None, label_map=None, key_suffix=""):
    """Evaluates a trained experiment on a split with the unified evaluator and stores the result in its record.
    mode: 'full' | 'sliced' | 'sliced+full'. Defaults: full-frame models -> 'full'; tile-trained -> 'sliced'."""
    rec = load_record(exp_id_or_spec) if isinstance(exp_id_or_spec, str) else load_record(exp_id_or_spec["experiment_id"])
    if rec is None:
        raise ValueError("experiment not trained/registered yet")
    spec = rec["spec"]
    mode = mode or ("sliced" if spec["tiling"] == "tile_train" else "full")
    res = int(res or spec["resolution"])
    if CFG.get("smoke") and CFG.get("smoke_max_res"):
        res = min(res, CFG["smoke_max_res"])
    if mode == "sliced+full":
        full_res = int(full_res or spec.get("full_res") or 1280)
        if CFG.get("smoke") and CFG.get("smoke_max_res"):
            full_res = min(full_res, CFG["smoke_max_res"])
    gt_name = gt_name or {"val": spec["val_set"], "test": spec["test_set"]}.get(split, split)
    pack = eval_pack or (spec["native_pack"] if mode.startswith("sliced") else spec["eval_pack"])
    key = f"{split}|{mode}|r{res}" + (f"+f{full_res}" if mode == "sliced+full" else "") + key_suffix
    if key in rec.get("evals", {}) and not force:
        print(f"[{rec['experiment_id']}] {key} already evaluated")
        return rec["evals"][key]
    pred = predictor or load_predictor(spec, rec["checkpoint_path"])
    dets = predict_dataset(pred, gt_name, pack, res, mode="full" if mode == "full" else "sliced",
                           tile=spec["tile"], overlap=spec["overlap"], full_res=full_res if mode == "sliced+full" else None)
    if label_map:  # e.g. MAVREC model evaluated on VisDrone mapped to MAVREC classes
        dets = [dict(d, category_id=label_map[d["category_id"]]) for d in dets if d["category_id"] in label_map]
    gt = json_path(gt_name)
    m = coco_evaluate(gt, dets, cat_ids=cat_ids)
    m.update(pr_at_best_f1(gt, dets))
    m.update(benchmark_predictor(pred, gt_name, pack, res, mode="full" if mode == "full" else "sliced",
                                 tile=spec["tile"], overlap=spec["overlap"], full_res=full_res if mode == "sliced+full" else None))
    m.update(dict(eval_split=split, inference_mode=mode, eval_resolution=res, full_res=full_res, gt=Path(gt).stem,
                  precision_mode=getattr(pred, "precision", "?"), evaluated=now()))
    if save_preds:
        pp = os.path.join(CFG["preds_dir"], f"{rec['experiment_id']}__{key.replace('|', '__').replace('+', '_')}.json.gz")
        save_json(dets, pp)
        m["preds_path"] = pp
    rec = load_record(rec["experiment_id"])
    rec.setdefault("evals", {})[key] = m
    if rec.get("parameters_M") is None and hasattr(pred, "n_params"):
        rec["parameters_M"] = round(pred.n_params() / 1e6, 2)
    save_record(rec)
    print(f"[{rec['experiment_id']}] {key}: mAP50-95={m['mAP50_95']} AP50={m['mAP50']} APS={m['APS']} APt={m['APt']} "
          f"R={m['recall']} FPS={m['FPS']}")
    del pred
    free_gpu()
    return m


def run_experiment(spec, eval_splits=("val",), eval_modes=None, force=False):
    """Train (resumable) + evaluate. Safe to re-run: finished parts are skipped."""
    exp_id = spec["experiment_id"]
    rec = load_record(exp_id)
    if rec and rec.get("status") == "done" and not force:
        missing = [(s, md) for s in eval_splits for md in (eval_modes or [None])
                   if not any(k.startswith(f"{s}|{md or ('sliced' if spec['tiling'] == 'tile_train' else 'full')}|")
                              for k in rec.get("evals", {}))]
        if not missing:
            print(f"[{exp_id}] done — skipping (force=True to redo)")
            return rec
    info = train(spec, force=force)
    rec = rec or {}
    rec.update({k: spec[k] for k in ["experiment_id", "stage", "model_family", "model_version", "backbone", "dataset_size",
                                    "train_set", "resolution", "tiling", "pretraining", "init", "epochs", "seed", "notes"]})
    rec.update(dict(batch_size=info["batch_size"], train_time_h=info["train_time_h"],
                    GPU_memory_train_GB=info["GPU_memory_train_GB"], checkpoint_path=info["weights"],
                    spec=spec, status="trained", lib_version=LIB_VERSION, comparable=True, smoke=CFG.get("smoke", False)))
    rec.setdefault("evals", {})
    rec.setdefault("created", now())
    save_record(rec)
    for s in eval_splits:
        for md in (eval_modes or [None]):
            evaluate(exp_id, split=s, mode=md)
    rec = load_record(exp_id)
    rec["status"] = "done"
    return save_record(rec)


def add_reference_record(exp_id, stage, model_version, metrics, notes, dataset_size=None, resolution=None,
                         split="val", protocol="reference"):
    """Historical (Review 2) or literature (paper) numbers. Marked comparable=False."""
    rec = dict(experiment_id=exp_id, stage=stage, model_family=MODEL_GROUP.get(model_version, "?"),
               model_version=model_version, backbone=MODEL_BACKBONE.get(model_version, "?"), dataset_size=dataset_size,
               resolution=resolution, tiling="none", pretraining="COCO", epochs=None, seed=None, status="reference",
               comparable=False, notes=notes, created=now(), lib_version=LIB_VERSION,
               evals={f"{split}|{protocol}|r{resolution}": dict(metrics, eval_split=split, inference_mode=protocol)})
    return save_record(rec)


SCHEMA = ["experiment_id", "stage", "model_family", "model_version", "backbone", "dataset_size", "resolution", "tiling",
          "inference_mode", "pretraining", "epochs", "batch_size", "seed", "eval_split", "eval_resolution",
          "mAP50", "mAP50_95", "AP75", "APt", "APS", "APM", "APL", "AR300", "mAP50_95_md100", "APS_md100",
          "precision", "recall", "F1", "conf_at_best_f1", "FPS", "latency_ms", "parameters_M", "GPU_memory_train_GB",
          "GPU_memory_infer_GB", "train_time_h", "checkpoint_path", "comparable", "status", "notes", "eval_key"]


def results_df(include_reference=True, stage=None):
    """Flatten every record x evaluation into one row (the master results table)."""
    import pandas as pd
    rows = []
    for p in sorted(glob.glob(os.path.join(CFG["records_dir"], "*.json"))):
        r = load_json(p)
        for k, ev in (r.get("evals") or {}).items():
            row = {c: r.get(c) for c in SCHEMA}
            row.update({c: ev.get(c) for c in SCHEMA if c in ev})
            row["eval_key"] = k
            row["per_class_AP"] = ev.get("per_class_AP")
            row["preds_path"] = ev.get("preds_path")
            rows.append(row)
    df = pd.DataFrame(rows, columns=SCHEMA + ["per_class_AP", "preds_path"]) if rows else pd.DataFrame(columns=SCHEMA)
    if not include_reference and len(df):
        df = df[df["comparable"] == True]  # noqa
    if stage and len(df):
        df = df[df["stage"].astype(str).str.startswith(stage)]
    return df.reset_index(drop=True)


def export_master_csv():
    df = results_df()
    p = os.path.join(CFG["tables_dir"], "master_results.csv")
    df.drop(columns=["per_class_AP"], errors="ignore").to_csv(p, index=False)
    print("Saved", p, df.shape)
    return df


# ----------------------------------------------------------------------------------
# Evidence-based selection
# ----------------------------------------------------------------------------------
SELECTION_ORDER = ["mAP50_95", "APS", "mAP50", "recall", "F1"]
EFFICIENCY_ORDER = [("FPS", False), ("GPU_memory_infer_GB", True), ("parameters_M", True)]


def select_best(df, tol=1.0, order=SELECTION_ORDER, verbose=True):
    """Lexicographic selection with a tolerance band (AP points):
    keep every candidate within `tol` of the best on metric 1; among those, within `tol` of the
    best on metric 2; ... ; remaining ties are broken by FPS (higher), inference memory (lower),
    parameter count (lower). Returns (best_row, explanation)."""
    c = df.dropna(subset=[order[0]]).copy()
    if not len(c):
        raise ValueError("no candidates with metrics")
    lines = [f"{len(c)} candidates: " + ", ".join(c["experiment_id"].astype(str))]
    for mtr in order:
        if c[mtr].isna().all():
            continue
        best = c[mtr].max()
        keep = c[c[mtr] >= best - tol]
        lines.append(f"{mtr}: best {best:.2f}; within {tol} -> {list(keep['experiment_id'])}")
        c = keep
        if len(c) == 1:
            break
    if len(c) > 1:
        for mtr, asc in EFFICIENCY_ORDER:
            if c[mtr].notna().any():
                c = c.sort_values(mtr, ascending=asc)
                lines.append(f"tie broken by {mtr} ({'lower' if asc else 'higher'} is better)")
                break
    best = c.iloc[0]
    lines.append(f"SELECTED: {best['experiment_id']}")
    expl = "\n".join(lines)
    if verbose:
        print(expl)
    return best, expl


def selection_table(df, cols=("mAP50_95", "APS", "mAP50", "recall", "precision", "F1", "FPS", "GPU_memory_infer_GB", "parameters_M")):
    """Multi-metric table with per-metric ranks (1 = best)."""
    t = df[["experiment_id", "model_family", "model_version", "resolution", "dataset_size"] + list(cols)].copy()
    for c in cols:
        asc = c in ("GPU_memory_infer_GB", "parameters_M")
        t[f"rank_{c}"] = t[c].rank(ascending=asc, method="min")
    return t.sort_values("mAP50_95", ascending=False).reset_index(drop=True)


def component_gain(base, new, tol=1.0):
    """Decide whether a component helps: +tol mAP50-95, or +tol APS without losing more than tol/2 mAP."""
    d_map = float(new["mAP50_95"]) - float(base["mAP50_95"])
    d_aps = float(new["APS"]) - float(base["APS"])
    keep = d_map >= tol or (d_aps >= tol and d_map >= -tol / 2)
    return dict(delta_mAP50_95=round(d_map, 2), delta_APS=round(d_aps, 2), keep=bool(keep))


def set_decision(key, value, reason=""):
    d = load_json(CFG["decisions_path"]) if os.path.exists(CFG["decisions_path"]) else {}
    d[key] = dict(value=value, reason=reason, set=now())
    save_json(d, CFG["decisions_path"], indent=1)
    print(f"DECISION {key} = {value}")
    return value


def get_decision(key, default=None, required=False):
    d = load_json(CFG["decisions_path"]) if os.path.exists(CFG["decisions_path"]) else {}
    if key in d:
        return d[key]["value"]
    if required and default is None:
        raise KeyError(f"Decision '{key}' not set yet — run the notebook that makes it (see README run order).")
    return default


def show_decisions():
    import pandas as pd
    d = load_json(CFG["decisions_path"]) if os.path.exists(CFG["decisions_path"]) else {}
    return pd.DataFrame([dict(key=k, **v) for k, v in d.items()])


def row_for(exp_id, split="val", mode=None):
    df = results_df()
    r = df[(df.experiment_id == exp_id) & (df.eval_split == split)]
    if mode:
        r = r[r.inference_mode == mode]
    return r.iloc[0] if len(r) else None


# ----------------------------------------------------------------------------------
# Visualisation helpers
# ----------------------------------------------------------------------------------
_COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
           (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190)]


def draw_boxes(img, boxes_xywh, labels1, scores=None, thickness=2, class_names=CLASS_NAMES):
    import cv2
    im = img.copy()
    for i, (b, c) in enumerate(zip(boxes_xywh, labels1)):
        x, y, w, h = [int(round(v)) for v in b]
        col = _COLORS[(c - 1) % len(_COLORS)]
        cv2.rectangle(im, (x, y), (x + w, y + h), col, thickness)
    return im


def show_gt(name, pack, n=4, seed=SEED, out_path=None):
    import matplotlib.pyplot as plt
    d = load_json(json_path(name))
    by = {}
    for a in d["annotations"]:
        by.setdefault(a["image_id"], []).append(a)
    ims = random.Random(seed).sample(d["images"], min(n, len(d["images"])))
    fig, axes = plt.subplots(1, len(ims), figsize=(6 * len(ims), 4))
    axes = np.atleast_1d(axes)
    for ax, im in zip(axes, ims):
        arr = read_rgb(image_path(pack, im["file_name"]))
        s = arr.shape[1] / im["width"]
        A = by.get(im["id"], [])
        ax.imshow(draw_boxes(arr, [[v * s for v in a["bbox"]] for a in A], [a["category_id"] for a in A], thickness=1))
        ax.set_title(f"{Path(im['file_name']).name}\n{len(A)} objects", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def prediction_grid(items, gt_name, pack, image_ids=None, n=3, conf=None, out_path=None, crop=None):
    """items: list of (title, preds_path, conf_threshold). Rows = images, cols = GT + each item."""
    import matplotlib.pyplot as plt
    d = load_json(json_path(gt_name))
    gt_by = {}
    for a in d["annotations"]:
        gt_by.setdefault(a["image_id"], []).append(a)
    ims = [im for im in d["images"] if (image_ids is None or im["id"] in image_ids)]
    if image_ids is None:
        ims = sorted(ims, key=lambda im: -len(gt_by.get(im["id"], [])))[:n]
    preds = []
    for title, pp, c in items:
        P = load_json(pp)
        by = {}
        for x in P:
            if x["score"] >= (c if c is not None else (conf or 0.25)):
                by.setdefault(x["image_id"], []).append(x)
        preds.append((title, by))
    cols = 1 + len(preds)
    fig, axes = plt.subplots(len(ims), cols, figsize=(5.5 * cols, 3.4 * len(ims)))
    axes = np.atleast_2d(axes)
    for r, im in enumerate(ims):
        arr = read_rgb(image_path(pack, im["file_name"]))
        s = arr.shape[1] / im["width"]
        G = gt_by.get(im["id"], [])
        panels = [("Ground truth", [[v * s for v in a["bbox"]] for a in G], [a["category_id"] for a in G])]
        for title, by in preds:
            P = by.get(im["id"], [])
            panels.append((title, [[v * s for v in p["bbox"]] for p in P], [p["category_id"] for p in P]))
        for c, (title, bx, lb) in enumerate(panels):
            vis = draw_boxes(arr, bx, lb, thickness=max(1, arr.shape[1] // 900))
            if crop:  # (y0, y1, x0, x1) as fractions of the image
                Hh, Ww = vis.shape[:2]
                vis = vis[int(crop[0] * Hh):int(crop[1] * Hh), int(crop[2] * Ww):int(crop[3] * Ww)]
            axes[r, c].imshow(vis)
            axes[r, c].set_title(f"{title} ({len(bx)})", fontsize=9)
            axes[r, c].axis("off")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def recovered_objects(gt_name, preds_a, conf_a, preds_b, conf_b, iou=0.5):
    """GT objects missed by A (e.g. full-frame) but found by B (e.g. tiling). Returns per-image counts
    and the list of recovered annotations."""
    gt = load_json(json_path(gt_name))
    A = [d for d in load_json(preds_a) if d["score"] >= conf_a]
    B = [d for d in load_json(preds_b) if d["score"] >= conf_b]
    _, _, ra = match_detections(gt, A, iou)
    _, _, rb = match_detections(gt, B, iou)
    rec = [ga for (ga, sa, _), (gb, sb, _) in zip(ra, rb) if sa < 0 <= sb]
    lost = [ga for (ga, sa, _), (gb, sb, _) in zip(ra, rb) if sb < 0 <= sa]
    cnt = {}
    for a in rec:
        cnt[a["image_id"]] = cnt.get(a["image_id"], 0) + 1
    return dict(recovered=rec, lost=lost, per_image=sorted(cnt.items(), key=lambda x: -x[1]))


def show_recovered(gt_name, pack, recovered, image_id, out_path=None, title="Recovered by tiling (green)"):
    import matplotlib.pyplot as plt, cv2
    d = load_json(json_path(gt_name))
    im = next(i for i in d["images"] if i["id"] == image_id)
    arr = read_rgb(image_path(pack, im["file_name"]))
    s = arr.shape[1] / im["width"]
    vis = arr.copy()
    for a in [r for r in recovered if r["image_id"] == image_id]:
        x, y, w, h = [int(v * s) for v in a["bbox"]]
        cv2.rectangle(vis, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), max(2, arr.shape[1] // 700))
    plt.figure(figsize=(14, 8))
    plt.imshow(vis); plt.axis("off"); plt.title(title)
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def savefig(name):
    import matplotlib.pyplot as plt
    p = os.path.join(CFG["figures_dir"], name)
    plt.savefig(p, dpi=200, bbox_inches="tight")
    print("saved", p)
    return p


def save_table(df, name):
    p = os.path.join(CFG["tables_dir"], name)
    df.to_csv(p + ".csv", index=False)
    try:
        Path(p + ".md").write_text(df.to_markdown(index=False))
    except Exception:
        pass
    print("saved", p + ".csv")
    return p


def status_board():
    """Quick overview of every experiment record (useful for team coordination)."""
    import pandas as pd
    rows = []
    for p in sorted(glob.glob(os.path.join(CFG["records_dir"], "*.json"))):
        r = load_json(p)
        rows.append(dict(experiment_id=r["experiment_id"], status=r.get("status"), evals=len(r.get("evals", {})),
                         train_time_h=r.get("train_time_h"), updated=r.get("updated")))
    for d in sorted(glob.glob(os.path.join(CFG["runs_dir"], "*"))):
        e = os.path.basename(d)
        if not any(x["experiment_id"] == e for x in rows):
            rows.append(dict(experiment_id=e, status="in progress / interrupted", evals=0,
                             train_time_h=load_json(os.path.join(d, "train_time.json"))["seconds"] / 3600
                             if os.path.exists(os.path.join(d, "train_time.json")) else None, updated=None))
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------------
# External (real-world) datasets: YOLO-format folders -> canonical COCO + local pack
# ----------------------------------------------------------------------------------
VISDRONE_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
# VisDrone class (1-based) -> MAVREC class id (1-based); unmapped classes are ignored in zero-shot evaluation
VISDRONE_TO_MAVREC = {1: 6, 2: 6, 3: 2, 4: 7, 5: 3, 6: 4, 9: 5}


def prepare_visdrone(yaml_name="VisDrone.yaml"):
    """Downloads/convert VisDrone2019-DET with Ultralytics' official dataset script (~2.3 GB) and
    returns dict(train=<images dir>, val=<images dir>, names=[...])."""
    from ultralytics.data.utils import check_det_dataset
    override = os.environ.get("MAVREC_VISDRONE_YAML")
    os.chdir(CFG["local_root"])
    info = check_det_dataset(override or yaml_name, autodownload=True)
    names = info["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    return dict(train=str(info["train"]), val=str(info["val"]), names=names)


def _yolo_dir_to_coco(images_dir, class_names, split, pack, max_images=None, seed=SEED):
    import cv2
    files = sorted(f for f in glob.glob(os.path.join(images_dir, "**", "*"), recursive=True) if f.lower().endswith(IMG_EXT))
    if max_images and len(files) > max_images:
        files = sorted(random.Random(seed).sample(files, max_images))
    images, anns = [], []
    pdir = pack_dir(pack)
    for f in files:
        lab = re.sub(r"([/\\])images([/\\])", r"\1labels\2", f, count=1)
        lab = os.path.splitext(lab)[0] + ".txt"
        try:
            from PIL import Image
            with Image.open(f) as _im:
                W, H = _im.size
        except Exception:
            continue
        fn = f"{split}/{os.path.basename(f)}"
        tgt = os.path.join(pdir, fn)
        os.makedirs(os.path.dirname(tgt), exist_ok=True)
        if not os.path.exists(tgt):
            os.symlink(f, tgt)
        iid = len(images) + 1
        images.append(dict(id=iid, file_name=fn, width=W, height=H, seq=sequence_id(os.path.basename(f)),
                           src_name=os.path.basename(f)))
        if os.path.exists(lab):
            for line in Path(lab).read_text().strip().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
                bw, bh = w * W, h * H
                if bw < 1 or bh < 1:
                    continue
                anns.append(dict(id=len(anns) + 1, image_id=iid, category_id=c + 1,
                                 bbox=[round((cx - w / 2) * W, 2), round((cy - h / 2) * H, 2), round(bw, 2), round(bh, 2)],
                                 area=round(bw * bh, 2), iscrowd=0))
    cats = [dict(id=i + 1, name=n, supercategory="object") for i, n in enumerate(class_names)]
    return dict(info=dict(source=images_dir, split=split), images=images, annotations=anns, categories=cats)


def register_yolo_dataset(name, train_images, val_images, class_names, train_sizes=(1000,), seed=SEED, val_max=None):
    """Creates ext/<name>_train_<n>.json (nested, seeded), ext/<name>_val.json and the local pack 'ext_<name>'.
    The pack holds symlinks, so it is rebuilt each session (cheap) after the dataset download."""
    pack = f"ext_{name}"
    os.makedirs(pack_dir(pack), exist_ok=True)
    full = _yolo_dir_to_coco(train_images, class_names, "train", pack)
    val = _yolo_dir_to_coco(val_images, class_names, "val", pack, max_images=val_max, seed=seed)
    save_json(val, os.path.join(CFG["ext_dir"], f"{name}_val.json"))
    order = list(full["images"])
    random.Random(seed).shuffle(order)
    out = {}
    for n in train_sizes:
        k = min(n, len(order))
        if k < n:
            print(f"  WARNING {name}: only {k} training images available for requested subset {n}")
        sub = subset_coco(full, [im["id"] for im in order[:k]], f"n{k}")
        save_json(sub, os.path.join(CFG["ext_dir"], f"{name}_train_{n}.json"))
        out[n] = f"ext/{name}_train_{n}"   # keyed by the REQUESTED size, which is what notebooks index with
    Path(pack_dir(pack), ".done").write_text(now())
    print(f"{name}: train pool {len(full['images'])} imgs; val {len(val['images'])} imgs / {len(val['annotations'])} boxes")
    return dict(train=out, val=f"ext/{name}_val", pack=pack, class_names=class_names)


def remap_gt_to_mavrec(src_name, mapping, out_name):
    """Builds a GT file in MAVREC label space (for zero-shot evaluation of MAVREC models)."""
    d = load_json(json_path(src_name))
    anns = [dict(a, category_id=mapping[a["category_id"]]) for a in d["annotations"] if a["category_id"] in mapping]
    cats = [dict(id=i + 1, name=n, supercategory="object") for i, n in enumerate(CLASS_NAMES)]
    out = dict(info=dict(d.get("info", {}), remapped=True), images=d["images"], annotations=anns, categories=cats)
    p = os.path.join(CFG["ext_dir"], out_name + ".json")
    save_json(out, p)
    return f"ext/{out_name}", sorted(set(mapping.values()))


def predict_folder(exp_id, folder, res=None, conf=0.25, out_dir=None, max_images=12, mode="full"):
    """Qualitative inference on any folder of images (e.g. flood UAV photos with no labels)."""
    import matplotlib.pyplot as plt
    rec = load_record(exp_id)
    spec = rec["spec"]
    pred = load_predictor(spec, rec["checkpoint_path"])
    res = res or spec["resolution"]
    files = sorted(f for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True) if f.lower().endswith(IMG_EXT))[:max_images]
    out_dir = out_dir or os.path.join(CFG["figures_dir"], f"qualitative_{exp_id}")
    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        arr = read_rgb(f)
        if mode == "full":
            b, s, l = pred.predict([arr], res)[0]
        else:
            b, s, l = sliced_predict(pred, arr, spec["tile"], spec["overlap"], tile_res=res)
        k = s >= conf
        bx = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in b[k]]
        vis = draw_boxes(arr, bx, (l[k] + 1).tolist(), thickness=max(1, arr.shape[1] // 800))
        plt.figure(figsize=(12, 7)); plt.imshow(vis); plt.axis("off")
        plt.title(f"{os.path.basename(f)} — {int(k.sum())} detections @conf {conf}")
        plt.savefig(os.path.join(out_dir, Path(f).stem + ".png"), dpi=120, bbox_inches="tight")
        plt.show()
    return out_dir


def combine_datasets(parts, out_name):
    """parts: list of (dataset_name, pack). Builds derived/<out_name>.json and a local symlink pack
    'combo_<out_name>' (used for the optional joint ground+aerial training). Re-run once per session."""
    pack = f"combo_{out_name}"
    pdir = pack_dir(pack)
    images, anns, cats = [], [], None
    for name, pk in parts:
        ensure_pack(pk)
        d = load_json(json_path(name))
        cats = d["categories"]
        idm = {}
        for im in d["images"]:
            fn = f"{pk}/{im['file_name']}"
            tgt = os.path.join(pdir, fn)
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            if not os.path.lexists(tgt):
                os.symlink(image_path(pk, im["file_name"]), tgt)
            idm[im["id"]] = len(images) + 1
            images.append(dict(im, id=idm[im["id"]], file_name=fn))
        for a in d["annotations"]:
            anns.append(dict(a, id=len(anns) + 1, image_id=idm[a["image_id"]]))
    Path(pdir, ".done").write_text(now())
    p = os.path.join(CFG["canon_dir"], "derived", out_name + ".json")
    save_json(dict(info=dict(combined=[x[0] for x in parts]), images=images, annotations=anns, categories=cats), p)
    print(f"combined dataset {out_name}: {len(images)} images, {len(anns)} boxes")
    return p, pack
