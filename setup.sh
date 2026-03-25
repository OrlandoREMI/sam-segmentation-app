#!/bin/bash
# setup.sh — Instala SAM2.1 + dependencias para RTX 2080
# Uso: bash setup.sh  (con conda env activado)
set -e

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         SAM2 Video Annotator — Setup                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Dependencias base ──────────────────────────────────────────────────────
echo "► Instalando dependencias Python..."
pip install gradio>=4.0 opencv-python pillow --quiet
pip install pycocotools --quiet && echo "  ✅ pycocotools OK" || echo "  ⚠️  pycocotools falló (se usará fallback de polígonos)"

# ── 2. SAM2 ───────────────────────────────────────────────────────────────────
echo ""
echo "► Clonando SAM2 (facebookresearch/sam2)..."

if [ -d "sam2_repo" ]; then
    echo "  sam2_repo ya existe — actualizando..."
    cd sam2_repo && git pull --quiet && cd ..
else
    git clone https://github.com/facebookresearch/sam2.git sam2_repo
fi

echo "► Instalando SAM2 (pip install -e .)..."
cd sam2_repo
pip install -e "." --quiet
echo "  ✅ SAM2 instalado"
cd ..

# ── 3. Checkpoints SAM2.1 (Sept 2024) ────────────────────────────────────────
echo ""
echo "► Descargando checkpoints SAM2.1..."
mkdir -p checkpoints
cd checkpoints

BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

download_ckpt() {
    local fname="$1"
    if [ -f "$fname" ]; then
        echo "  ✅ $fname ya existe"
    else
        echo "  ⬇️  Descargando $fname..."
        wget -q --show-progress "${BASE_URL}/${fname}" -O "$fname"
        echo "  ✅ $fname descargado"
    fi
}

download_ckpt "sam2.1_hiera_small.pt"
download_ckpt "sam2.1_hiera_base_plus.pt"
download_ckpt "sam2.1_hiera_large.pt"
cd ..

# ── 4. Verificar configs ──────────────────────────────────────────────────────
# Los YAMLs NO se copian manualmente — SAM2 los resuelve desde el paquete
# instalado en sam2_repo/sam2/configs/sam2.1/ vía Hydra. No hay que tocarlos.
echo ""
echo "► Verificando configs SAM2.1 en el paquete..."
python3 -c "
import sam2, os
base = os.path.dirname(sam2.__file__)
cfg_dir = os.path.join(base, 'configs', 'sam2.1')
cfgs = ['sam2.1_hiera_s.yaml', 'sam2.1_hiera_b+.yaml', 'sam2.1_hiera_l.yaml']
all_ok = True
for c in cfgs:
    if os.path.exists(os.path.join(cfg_dir, c)):
        print(f'  OK {c}')
    else:
        print(f'  MISSING {c}  (dir: {cfg_dir})')
        all_ok = False
exit(0 if all_ok else 1)
" && echo "  ✅ Configs OK" || echo "  ⚠️  Algún config falta — revisa la instalación"

# ── 5. Carpeta de salida ──────────────────────────────────────────────────────
mkdir -p outputs

# ── 6. Resumen ────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Setup completado                                        ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Para iniciar la app:                                    ║"
echo "║                                                          ║"
echo "║      python app.py                                       ║"
echo "║                                                          ║"
echo "║  Abrir en navegador:  http://localhost:7860              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

python3 -c "
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'  GPU: {p.name}  ({p.total_memory/1e9:.1f} GB VRAM)')
    if p.total_memory < 10e9:
        print('  INFO 8 GB — Small y Base+ son seguros; Large al limite')
    else:
        print('  OK VRAM suficiente para cualquier checkpoint')
else:
    print('  WARN No se detecto GPU CUDA — se usara CPU (muy lento)')
" 2>/dev/null || true

echo ""
