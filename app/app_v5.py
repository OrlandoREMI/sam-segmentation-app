#!/usr/bin/env python3
"""
SAM2 Video Annotator — Bardas & Rejas  v5
──────────────────────────────────────────
Click sobre barda/reja → SAM2 propaga la máscara al video completo.

Recuperación de sesión:
    Los clicks se guardan en session_points.json tras cada click.
    Si la app se cierra o el browser reconecta, extrae los frames
    y pulsa "Restaurar sesión" — los puntos se recuperan sin reanotar.
"""

import gc, os, cv2, json, shutil, torch, warnings
import numpy as np
import gradio as gr
from pathlib import Path
from PIL import Image
from datetime import datetime

warnings.filterwarnings("ignore")

# ── SAM2 ──────────────────────────────────────────────────────────────────────
try:
    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("⚠️  sam2 no instalado. Corre setup.sh primero.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    _cc   = torch.cuda.get_device_capability()
    DTYPE = torch.bfloat16 if _cc[0] >= 8 else torch.float16
else:
    DTYPE = torch.float32

MODELS = {
    "🟢 Small  — rápido, seguro 8 GB VRAM": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "checkpoints/sam2.1_hiera_small.pt",
    ),
    "🟡 Base+  — balance calidad/velocidad": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "checkpoints/sam2.1_hiera_base_plus.pt",
    ),
    "🔴 Large  — máxima calidad (~7-8 GB)": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "checkpoints/sam2.1_hiera_large.pt",
    ),
}

SESSION_FILE = Path("session_points.json")
VIDEO_DIR    = Path(os.environ.get("SAM2_VIDEO_DIR", "videos"))

# ── Auth / session lock ───────────────────────────────────────────────────────
USERS = {
    "orlando": "bitxo4-dafbiq-toqbYr",
    "alan":    "bupvan-xedhoW-gadsi0",
    "luis":    "xensap-gUwsev-zerka5",
}
SESSION_TIMEOUT_MIN = 30

import time as _time
_app_lock = dict(user=None, since=None)

def _lock_owner():
    if _app_lock["user"] is None:
        return None
    if (_time.time() - _app_lock["since"]) / 60 > SESSION_TIMEOUT_MIN:
        _app_lock["user"] = _app_lock["since"] = None
        return None
    return _app_lock["user"]

def _touch_lock(username: str):
    _app_lock["user"]  = username
    _app_lock["since"] = _time.time()

def _release_lock():
    _app_lock["user"] = _app_lock["since"] = None

def _auth_fn(username: str, password: str) -> bool:
    if USERS.get(username) != password:
        return False
    owner = _lock_owner()
    if owner is None or owner == username:
        _touch_lock(username)
        return True
    return False

# ── Constantes ────────────────────────────────────────────────────────────────
OBJ_ID     = 1
MASK_COLOR = np.array([50, 220, 120], dtype=np.uint8)
ALPHA      = 0.45

# ── Estado global ─────────────────────────────────────────────────────────────
S = dict(
    predictor    = None,
    loaded_model = None,
    inf_state    = None,
    frames       = [],
    frame_dir    = None,
    current      = 0,
    all_points   = {},
    masks        = {},
    fps          = 30.0,
    video_path   = None,
    stride       = 2,
    max_dim      = 1280,
)

# ── Gestión de VRAM ───────────────────────────────────────────────────────────

def _free_vram():
    """
    Libera TODA la memoria GPU del estado actual.
    Llamar ANTES de cualquier operación que aloje nuevos tensores grandes.
    El orden importa:
      1. Liberar inf_state (tensores de frames + features)
      2. Liberar predictor (pesos del modelo)
      3. gc.collect() para que Python libere referencias inmediatamente
      4. empty_cache() para devolver la memoria al pool CUDA
    """
    S["inf_state"] = None
    S["predictor"] = None
    if DEVICE == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

def _free_inf_state():
    """Libera solo el estado de inferencia (frames en GPU), conserva el modelo."""
    S["inf_state"] = None
    if DEVICE == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

# ── Persistencia ──────────────────────────────────────────────────────────────

def _save_points():
    data = {
        "video_path": S["video_path"],
        "stride":     S["stride"],
        "max_dim":    S["max_dim"],
        "points":     {str(k): v for k, v in S["all_points"].items()},
    }
    try:
        SESSION_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"⚠️  No se pudo guardar sesión: {e}")

def _load_session_file():
    if not SESSION_FILE.exists():
        return None
    try:
        return json.loads(SESSION_FILE.read_text())
    except Exception:
        return None

def scan_videos() -> list[str]:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    exts  = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}
    files = sorted([f.name for f in VIDEO_DIR.iterdir() if f.suffix in exts])
    return files if files else []

def _model_is_alive() -> bool:
    """Verifica que el predictor existe y tiene parámetros accesibles en VRAM.
    No copia datos — solo accede al primer parámetro para confirmar que está vivo.
    """
    if S["predictor"] is None:
        return False
    try:
        next(S["predictor"].parameters())
        return True
    except Exception:
        return False


def restore_session():
    data = _load_session_file()
    if data is None:
        return overlay_frame(S["current"]), "⚠️  No hay sesión guardada"
    if not S["frames"]:
        return None, "⚠️  Extrae los frames del video primero, luego restaura"

    S["all_points"] = {
        int(k): [tuple(p) for p in v]
        for k, v in data.get("points", {}).items()
    }
    S["masks"].clear()
    # Solo restaura puntos en RAM. SAM2 se inicializa limpio al propagar.

    n = len(S["all_points"])
    saved_video = data.get("video_path", "?")
    return overlay_frame(S["current"]), (
        f"✅ {n} frames restaurados · "
        f"Video: {Path(saved_video).name if saved_video else '?'} · "
        f"Listo para propagar"
    )

def on_page_load():
    """
    Se dispara cada vez que el browser carga la página (incluyendo reconexiones).
    Repuebla la UI con el estado actual del proceso Python:
      - Canvas: frame actual si hay frames en RAM
      - Status modelo: vivo / necesita recarga / no cargado
      - Status video: frames disponibles o vacío
      - Status sesión: puntos en memoria
    """
    # Canvas y puntos — viven en RAM, siempre recuperables
    canvas_val  = overlay_frame(S["current"]) if S["frames"] else None

    # Status del modelo — verificar VRAM honestamente
    if _model_is_alive():
        model_val = f"✅ {S['loaded_model']}{vram_info()}"
    elif S["loaded_model"] is not None:
        # loaded_model tiene nombre pero el predictor no responde → se liberó
        model_val = f"⚠️  {S['loaded_model']} — no está en VRAM, recarga el modelo"
        S["predictor"]    = None
        S["loaded_model"] = None
    else:
        model_val = "Sin modelo cargado"

    # Status del video
    if S["frames"]:
        h, w    = S["frames"][0].shape[:2]
        vid_val = f"✅ {len(S['frames'])} frames en memoria · {w}×{h}"
    else:
        vid_val = "Sin frames en memoria"

    # Status de sesión / puntos
    n_anotados = sum(1 for pts in S["all_points"].values() if pts)
    if n_anotados:
        session_val = f"✅ {n_anotados} frames con puntos en memoria"
    else:
        # Chequear si hay sesión guardada en disco aunque no esté en memoria
        data = _load_session_file()
        if data:
            n = len(data.get("points", {}))
            v = Path(data.get("video_path", "?")).name
            session_val = f"💾 Sesión en disco: {n} frames · {v} — pulsa Restaurar"
        else:
            session_val = ""

    return canvas_val, model_val, vid_val, session_val


# ── Helpers visualización ─────────────────────────────────────────────────────

def vram_info():
    if DEVICE == "cuda":
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f" · VRAM {used:.1f}/{total:.1f} GB"
    return " · CPU"

def overlay_frame(idx: int):
    if not S["frames"]:
        return None
    idx   = max(0, min(idx, len(S["frames"]) - 1))
    frame = S["frames"][idx].copy()

    if idx in S["masks"] and S["masks"][idx].any():
        m       = S["masks"][idx]
        overlay = frame.copy()
        overlay[m] = (overlay[m] * (1 - ALPHA) + MASK_COLOR * ALPHA).astype(np.uint8)
        contours, _ = cv2.findContours(
            m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, tuple(MASK_COLOR.tolist()), 2)
        frame = overlay

    for (x, y, label) in S["all_points"].get(idx, []):
        color = (80, 255, 80) if label == 1 else (255, 60, 60)
        cv2.circle(frame, (int(x), int(y)), 9, color, -1)
        cv2.circle(frame, (int(x), int(y)), 9, (255, 255, 255), 2)

    return Image.fromarray(frame)

# ── SAM2 state ────────────────────────────────────────────────────────────────

def _rebuild_sam2_state():
    """
    Único punto donde se crea un inf_state nuevo.
    Libera el anterior antes de alojar el nuevo para evitar OOM.
    Solo se llama desde propagate_and_export.
    """
    if S["predictor"] is None or S["frame_dir"] is None:
        return
    _free_inf_state()
    S["inf_state"] = S["predictor"].init_state(video_path=S["frame_dir"])
    for fidx, pts in S["all_points"].items():
        if not pts:
            continue
        pts_arr = np.array([[x, y] for x, y, _ in pts], dtype=np.float32)
        lbl_arr = np.array([l       for _, _, l in pts], dtype=np.int32)
        with torch.autocast(DEVICE, dtype=DTYPE):
            S["predictor"].add_new_points_or_box(
                inference_state=S["inf_state"],
                frame_idx=fidx,
                obj_id=OBJ_ID,
                points=pts_arr,
                labels=lbl_arr,
            )

def _sam2_preview(idx: int):
    """Preview de máscara en frame idx. No afecta propagación final."""
    if S["inf_state"] is None:
        return
    pts = S["all_points"].get(idx)
    if not pts:
        return
    pts_arr = np.array([[x, y] for x, y, _ in pts], dtype=np.float32)
    lbl_arr = np.array([l       for _, _, l in pts], dtype=np.int32)
    with torch.autocast(DEVICE, dtype=DTYPE):
        _, _, logits = S["predictor"].add_new_points_or_box(
            inference_state=S["inf_state"],
            frame_idx=idx,
            obj_id=OBJ_ID,
            points=pts_arr,
            labels=lbl_arr,
        )
    S["masks"][idx] = (logits[0, 0] > 0).cpu().numpy()

# ── Acciones ──────────────────────────────────────────────────────────────────

def load_model(model_key: str) -> str:
    if not SAM2_AVAILABLE:
        return "❌ sam2 no instalado — corre: bash setup.sh"
    cfg, ckpt = MODELS[model_key]
    if not Path(ckpt).exists():
        return f"❌ Checkpoint no encontrado: {ckpt}"
    if S["loaded_model"] == model_key and S["predictor"] is not None:
        return f"✅ {model_key} ya estaba cargado{vram_info()}"

    # Liberar modelo e inf_state anteriores completamente antes de cargar
    _free_vram()
    S["loaded_model"] = None

    S["predictor"]    = build_sam2_video_predictor(cfg, ckpt, device=DEVICE)
    S["loaded_model"] = model_key

    # Si hay frames ya extraídos, inicializar inf_state
    if S["frame_dir"] and Path(S["frame_dir"]).exists():
        S["inf_state"] = S["predictor"].init_state(video_path=S["frame_dir"])

    return f"✅ Modelo cargado{vram_info()}"


def load_video(video_name: str, stride: int = 2, max_dim: int = 1280):
    if not video_name:
        return None, "⚠️  Selecciona un video primero"
    video_path = str(VIDEO_DIR / video_name)
    if not Path(video_path).exists():
        return None, f"❌ No se encontró: {video_path}"

    stride  = max(1, int(stride))
    max_dim = int(max_dim)

    # Liberar inf_state anterior (no el modelo) y frames de RAM
    _free_inf_state()
    if S["frame_dir"] and Path(S["frame_dir"]).exists():
        shutil.rmtree(S["frame_dir"])

    S.update(
        frames=[], masks={}, all_points={}, current=0,
        inf_state=None, video_path=video_path, stride=stride, max_dim=max_dim,
    )

    frame_dir = Path("/tmp/sam2_frames") / datetime.now().strftime("%H%M%S%f")
    frame_dir.mkdir(parents=True)
    S["frame_dir"] = str(frame_dir)

    cap       = cv2.VideoCapture(video_path)
    S["fps"]  = (cap.get(cv2.CAP_PROP_FPS) or 30.0) / stride
    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_idx   = saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if raw_idx % stride == 0:
            if max_dim > 0:
                h, w = frame.shape[:2]
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    frame = cv2.resize(
                        frame, (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            S["frames"].append(rgb)
            cv2.imwrite(
                str(frame_dir / f"{saved_idx:05d}.jpg"), frame,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            saved_idx += 1
        raw_idx += 1
    cap.release()

    n = len(S["frames"])
    if n == 0:
        return None, "❌ No se pudieron leer frames del video"

    h, w = S["frames"][0].shape[:2]

    # Inicializar inf_state si el modelo ya está cargado
    if S["predictor"] is not None:
        S["inf_state"] = S["predictor"].init_state(video_path=str(frame_dir))

    return overlay_frame(0), (
        f"✅ {n} frames de {total_raw} · stride {stride} · "
        f"{S['fps']:.1f} FPS · {w}×{h}\n"
        f"💾 Pulsa 'Restaurar sesión' si tienes anotaciones guardadas"
    )


def on_click(evt: gr.SelectData, neg_mode: bool):
    if evt is None or not hasattr(evt, "index") or evt.index is None:
        return overlay_frame(S["current"]), "⚠️  Click fuera de la imagen"
    _touch_lock(_lock_owner() or "")
    if not S["frames"]:
        return None, "⚠️  Extrae los frames primero"
    if S["predictor"] is None:
        return None, "⚠️  Carga un modelo primero"
    if S["inf_state"] is None:
        return None, "⚠️  Estado SAM2 no inicializado — re-extrae los frames"

    try:
        x, y = int(evt.index[0]), int(evt.index[1])
    except Exception as e:
        return overlay_frame(S["current"]), f"⚠️  Error leyendo click: {e}"

    label = 0 if neg_mode else 1
    idx   = S["current"]
    S["all_points"].setdefault(idx, []).append((x, y, label))
    _save_points()
    _sam2_preview(idx)

    n_pos = sum(1 for *_, l in S["all_points"].get(idx, []) if l == 1)
    n_neg = sum(1 for *_, l in S["all_points"].get(idx, []) if l == 0)
    total = sum(1 for pts in S["all_points"].values() if pts)
    mode  = "negativo 🔴" if neg_mode else "positivo 🟢"
    return overlay_frame(idx), (
        f"Frame {idx}/{len(S['frames'])-1} · Click {mode} ({x},{y}) · "
        f"+{n_pos}/-{n_neg} · {total} frames anotados · 💾"
    )


def clear_current_frame():
    idx = S["current"]
    S["all_points"].pop(idx, None)
    S["masks"].pop(idx, None)
    _save_points()
    return overlay_frame(idx), f"Frame {idx} · puntos borrados · 💾"


def clear_all():
    S["all_points"].clear()
    S["masks"].clear()
    _save_points()
    return overlay_frame(S["current"]), "🗑️  Todo borrado · 💾"


def navigate(delta: int):
    n = len(S["frames"])
    if n == 0:
        return None, "Sin frames cargados"
    S["current"] = max(0, min(S["current"] + delta, n - 1))
    idx  = S["current"]
    pts  = len(S["all_points"].get(idx, []))
    mask = "✅ máscara" if idx in S["masks"] and S["masks"][idx].any() else "sin máscara"
    return overlay_frame(idx), f"Frame {idx}/{n-1} · {pts} puntos · {mask}"


def goto_frame(idx_val):
    try:
        idx = int(idx_val)
    except (ValueError, TypeError):
        idx = S["current"]
    n = len(S["frames"])
    if n == 0:
        return None, "Sin frames cargados"
    S["current"] = max(0, min(idx, n - 1))
    pts  = len(S["all_points"].get(S["current"], []))
    mask = "✅ máscara" if S["current"] in S["masks"] and S["masks"][S["current"]].any() else "sin máscara"
    return overlay_frame(S["current"]), f"Frame {S['current']}/{n-1} · {pts} puntos · {mask}"


def propagate_and_export(exp_frames: bool, exp_video: bool, exp_coco: bool):
    _touch_lock(_lock_owner() or "")
    if not S["frames"]:
        return None, "⚠️  Extrae los frames primero"
    if S["predictor"] is None:
        return None, "⚠️  Carga un modelo primero"
    if not any(pts for pts in S["all_points"].values()):
        return None, "⚠️  Agrega al menos un click positivo antes de propagar"

    # Estado limpio: libera inf_state anterior, construye uno nuevo con puntos limpios
    _rebuild_sam2_state()

    out_dir = Path("outputs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)

    S["masks"].clear()
    with torch.autocast(DEVICE, dtype=DTYPE):
        for frame_idx, _, mask_logits in S["predictor"].propagate_in_video(S["inf_state"]):
            S["masks"][frame_idx] = (mask_logits[0, 0] > 0).cpu().numpy()

    n_masked = sum(1 for m in S["masks"].values() if m.any())
    results  = [f"✅ {n_masked}/{len(S['frames'])} frames con máscara{vram_info()}"]

    if exp_frames:
        fd = out_dir / "frames"; fd.mkdir()
        md = out_dir / "masks";  md.mkdir()
        for i, frame in enumerate(S["frames"]):
            Image.fromarray(frame).save(fd / f"{i:05d}.jpg", quality=95)
            mask_arr = S["masks"].get(i, np.zeros(frame.shape[:2], bool))
            Image.fromarray((mask_arr * 255).astype(np.uint8)).save(md / f"{i:05d}.png")
        results.append(f"📁 frames/ + masks/  →  {out_dir}")

    if exp_video:
        vpath  = str(out_dir / "overlay.mp4")
        h, w   = S["frames"][0].shape[:2]
        writer = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), S["fps"], (w, h))
        for i, frame in enumerate(S["frames"]):
            vis = frame.copy()
            m   = S["masks"].get(i)
            if m is not None and m.any():
                vis[m] = (vis[m] * (1 - ALPHA) + MASK_COLOR * ALPHA).astype(np.uint8)
                contours, _ = cv2.findContours(
                    m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(vis, contours, -1, tuple(MASK_COLOR.tolist()), 2)
            writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        writer.release()
        results.append(f"🎥 overlay.mp4  →  {vpath}")

    if exp_coco:
        _export_coco(out_dir)
        results.append(f"📋 annotations.json  →  {out_dir}")

    return overlay_frame(S["current"]), "\n".join(results)


def release_app():
    _release_lock()
    return ""


def _export_coco(out_dir: Path):
    try:
        from pycocotools import mask as coco_mask
        use_rle = True
    except ImportError:
        use_rle = False

    images, annotations, ann_id = [], [], 1
    for i, frame in enumerate(S["frames"]):
        h, w = frame.shape[:2]
        images.append({"id": i, "file_name": f"{i:05d}.jpg", "height": h, "width": w})
        m = S["masks"].get(i)
        if m is None or not m.any():
            continue
        if use_rle:
            rle           = coco_mask.encode(np.asfortranarray(m.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")
            area          = float(coco_mask.area(rle))
            bbox          = list(map(float, coco_mask.toBbox(rle)))
            seg           = rle
        else:
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            seg  = [c.flatten().tolist() for c in contours if len(c) >= 3]
            area = float(m.sum())
            ys, xs = np.where(m)
            bbox = [float(xs.min()), float(ys.min()),
                    float(xs.max()-xs.min()), float(ys.max()-ys.min())]
            if not seg:
                continue
        annotations.append({
            "id": ann_id, "image_id": i, "category_id": 1,
            "segmentation": seg, "area": area, "bbox": bbox, "iscrowd": 0,
        })
        ann_id += 1

    coco = {
        "info":        {"description": "SAM2 Fence/Wall Annotations", "version": "1.0"},
        "categories":  [{"id": 1, "name": "fence_wall", "supercategory": "barrier"}],
        "images":      images,
        "annotations": annotations,
    }
    with open(out_dir / "annotations.json", "w") as f:
        json.dump(coco, f, indent=2)

# ── UI ────────────────────────────────────────────────────────────────────────
CSS = """
footer { display: none !important; }
.compact-group { padding: 8px !important; }
"""

with gr.Blocks(
    title="SAM2 Fence Annotator",
    css=CSS,
    theme=gr.themes.Base(primary_hue="emerald", neutral_hue="slate"),
) as demo:

    gr.Markdown(
        "# 🛸 SAM2 Fence Annotator · Bardas & Rejas  "
        f"· `{DEVICE.upper()}` · "
        f"`{'float16' if DTYPE == torch.float16 else 'bfloat16' if DTYPE == torch.bfloat16 else 'float32'}` · "
        f"{'✅ SAM2' if SAM2_AVAILABLE else '❌ SAM2 no instalado'}"
    )

    # ── Fila principal: columna izquierda + canvas ────────────────────────────
    with gr.Row(equal_height=False):

        # Columna izquierda delgada
        with gr.Column(scale=1, min_width=260):

            with gr.Group():
                gr.Markdown("#### ⚙️ Modelo")
                model_dd       = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value=list(MODELS.keys())[0],
                    show_label=False,
                )
                load_model_btn = gr.Button("Cargar modelo", variant="primary", size="sm")
                model_status   = gr.Textbox(
                    interactive=False, show_label=False,
                    placeholder="Sin modelo cargado...", lines=1,
                )

            with gr.Group():
                gr.Markdown("#### 📹 Video")
                with gr.Row():
                    video_dd    = gr.Dropdown(
                        choices=scan_videos(), value=None,
                        label=f"{VIDEO_DIR}/", show_label=True, scale=5,
                    )
                    refresh_btn = gr.Button("🔄", scale=1, size="sm", min_width=36)
                with gr.Row():
                    stride_sl = gr.Slider(
                        minimum=1, maximum=10, value=2, step=1,
                        label="Stride", info="1 de cada N frames",
                    )
                    maxdim_sl = gr.Slider(
                        minimum=0, maximum=1920, value=1280, step=64,
                        label="Res. máx", info="px (0=original)",
                    )
                load_vid_btn = gr.Button("Extraer frames", variant="primary", size="sm")
                vid_status   = gr.Textbox(
                    interactive=False, show_label=False,
                    placeholder="Sin video cargado...", lines=2,
                )

            with gr.Group():
                gr.Markdown("#### 💾 Sesión")
                restore_btn     = gr.Button("Restaurar sesión guardada", variant="secondary", size="sm")
                logout_btn      = gr.Button("🚪 Cerrar sesión", variant="secondary", size="sm")
                session_info    = gr.Textbox(interactive=False, show_label=False,
                                             placeholder="Sin sesión activa...", lines=2)
                logout_redirect = gr.HTML(value="", visible=True)

        # Canvas grande
        with gr.Column(scale=3):
            canvas = gr.Image(
                label="Frame actual — haz click para anotar",
                type="pil", interactive=True, height=620,
            )
            status_box = gr.Textbox(
                interactive=False, show_label=False,
                placeholder="Esperando acción...", lines=2,
            )

    # ── Fila inferior: anotación · navegación · exportar ─────────────────────
    with gr.Row():

        with gr.Column(scale=1):
            gr.Markdown("#### 🖱️ Anotación")
            neg_toggle = gr.Checkbox(label="🔴 Modo negativo (excluir)", value=False)
            with gr.Row():
                clear_frame_btn = gr.Button("🗑 Frame actual", size="sm")
                clear_all_btn   = gr.Button("🗑 Todo", size="sm", variant="stop")

        with gr.Column(scale=1):
            gr.Markdown("#### ↔️ Navegación")
            with gr.Row():
                prev_btn = gr.Button("◀ Anterior", size="sm")
                next_btn = gr.Button("Siguiente ▶", size="sm")
            with gr.Row():
                frame_input = gr.Number(label="Frame #", value=0, precision=0, scale=3)
                goto_btn    = gr.Button("Ir", size="sm", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("#### 📤 Exportar")
            exp_frames_cb = gr.Checkbox(label="Frames RGB + máscaras PNG", value=True)
            exp_video_cb  = gr.Checkbox(label="Video con overlay", value=True)
            exp_coco_cb   = gr.Checkbox(label="Anotaciones COCO JSON", value=False)
            propagate_btn = gr.Button("🚀 Propagar y exportar", variant="primary")

    # ── Eventos ───────────────────────────────────────────────────────────────
    demo.load(
        fn=on_page_load,
        inputs=None,
        outputs=[canvas, model_status, vid_status, session_info],
    )
    load_model_btn.click(load_model, inputs=model_dd, outputs=model_status)
    load_vid_btn.click(load_video, inputs=[video_dd, stride_sl, maxdim_sl],
                       outputs=[canvas, vid_status])
    refresh_btn.click(fn=lambda: gr.update(choices=scan_videos()), outputs=video_dd)
    restore_btn.click(restore_session, outputs=[canvas, session_info])
    logout_btn.click(
        fn=release_app, inputs=None, outputs=logout_redirect,
        js="async () => { window.location.href = '/logout'; }",
    )
    canvas.select(fn=on_click, inputs=[neg_toggle], outputs=[canvas, status_box])
    clear_frame_btn.click(clear_current_frame, outputs=[canvas, status_box])
    clear_all_btn.click(clear_all,             outputs=[canvas, status_box])
    prev_btn.click(lambda: navigate(-1), outputs=[canvas, status_box])
    next_btn.click(lambda: navigate(+1), outputs=[canvas, status_box])
    goto_btn.click(goto_frame, inputs=frame_input, outputs=[canvas, status_box])
    propagate_btn.click(
        propagate_and_export,
        inputs=[exp_frames_cb, exp_video_cb, exp_coco_cb],
        outputs=[canvas, status_box],
    )

if __name__ == "__main__":
    print(f"Device : {DEVICE}")
    dtype_name = 'bfloat16' if DTYPE == torch.bfloat16 else 'float16' if DTYPE == torch.float16 else 'float32'
    print(f"dtype  : {dtype_name}")
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"GPU    : {p.name}  ({p.total_memory/1e9:.1f} GB VRAM)")
    data = _load_session_file()
    if data:
        n = len(data.get("points", {}))
        print(f"💾 Sesión guardada: {n} frames · Video: {data.get('video_path','?')}")
        print(f"   Extrae los frames y pulsa 'Restaurar sesión'")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        auth=_auth_fn,
        auth_message="Verifica que no haya otro usuario haciendo uso de la app.",
        share=False,
        inbrowser=True,
    )
