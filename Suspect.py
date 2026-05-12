import cv2
import json
import os

def generate_coco_from_folder(input_folder, output_json_path):
    """
    Recorre una carpeta de máscaras, detecta blobs y genera un único JSON COCO.
    """
    coco_data = {
        "info": {"description": "Dataset generado por script", "version": "1.0"},
        "licenses": [],
        "categories": [{"id": 2, "name": "suspect", "supercategory": "shape"}],
        "images": [],
        "suspect": []
    }
    
    # Contadores globales
    image_id = 1
    annotation_id = 1
    
    # Extensiones válidas para imágenes
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    
    print(f"Procesando {len(files)} imágenes...")

    for filename in files:
        file_path = os.path.join(input_folder, filename)
        
        # 1. Cargar máscara
        mask = cv2.imread(file_path, 0)
        if mask is None:
            continue
            
        height, width = mask.shape
        
        name_without_ext = os.path.splitext(filename)[0]
        new_filename = f"{name_without_ext}.jpg"
        
        # 2. Agregar información de la imagen al JSON
        coco_data["images"].append({
            "id": image_id,
            "file_name": new_filename,
            "width": width,
            "height": height
        })
        
        # 3. Detectar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 4. Procesar contornos y agregar a anotaciones
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            coco_data["suspect"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0
            })
            annotation_id += 1
        
        image_id += 1
        print(f"  > Procesado: {filename}")

    # 5. Guardar el archivo JSON final
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)
        
    print(f"\nÉxito. Se ha generado '{output_json_path}' con {len(coco_data['suspect'])} anotaciones totales.")

# --- Uso del script ---
# Asegúrate de poner la ruta de tu carpeta aquí:
generate_coco_from_folder('/home/alan02/Documentos/masks2', 'dataset_suspect1.json')
