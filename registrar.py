import os
import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace

# ================================
# CONFIGURACIÓN
# ================================

# Ruta donde se guardarán los vectores
VECTOR_DB_PATH = "/home/simonperez/GIT/Reconocimiento-facial/database/"
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# Modelo de embeddings por defecto (🔹 siempre DeepID)
MODEL_NAME = "DeepID"

# Inicializar MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh

# ================================
# FUNCIONES
# ================================

def obtener_rostro(filepath):
    """Recorta el rostro usando MediaPipe FaceMesh"""
    img = cv2.imread(filepath)
    if img is None:
        print(f"❌ No se pudo leer la imagen {filepath}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            print(f"⚠️ No se detectó rostro en {filepath}")
            return None

        h, w, _ = img.shape
        x_coords = [int(lm.x * w) for lm in results.multi_face_landmarks[0].landmark]
        y_coords = [int(lm.y * h) for lm in results.multi_face_landmarks[0].landmark]

        x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
        y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))

        # Recorte del rostro
        rostro = img_rgb[y_min:y_max, x_min:x_max]
        return rostro


def generar_vector(filepath, nombre):
    """Genera el vector de embedding DeepID y lo guarda en la base de datos"""
    rostro = obtener_rostro(filepath)
    if rostro is None:
        return False

    tmp_path = "tmp_face.jpg"
    cv2.imwrite(tmp_path, cv2.cvtColor(rostro, cv2.COLOR_RGB2BGR))

    try:
        rep = DeepFace.represent(img_path=tmp_path,
                                 model_name=MODEL_NAME,
                                 enforce_detection=False)

        if not rep or "embedding" not in rep[0]:
            print(f"❌ No se pudo generar embedding para {nombre}")
            return False

        embedding = np.array(rep[0]["embedding"])

        save_path = os.path.join(VECTOR_DB_PATH, f"{nombre}.npy")

        # Evitar sobrescribir si ya existe
        if os.path.exists(save_path):
            print(f"⚠️ El vector de {nombre} ya existe en {save_path}, no se sobrescribió.")
            return True

        np.save(save_path, embedding)
        print(f"✅ Vector de {nombre} guardado en {save_path} con modelo {MODEL_NAME}")
        return True

    except Exception as e:
        print(f"❌ Error al generar embedding para {nombre}: {e}")
        return False

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ================================
# PROCESAMIENTO DE ESTUDIANTES
# ================================

IMAGES_FOLDER = "/home/simonperez/GIT/Reconocimiento-facial/data/estudiantes"

if not os.path.exists(IMAGES_FOLDER):
    print(f"❌ La carpeta {IMAGES_FOLDER} no existe")
else:
    archivos = [f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    if not archivos:
        print(f"⚠️ No se encontraron imágenes en {IMAGES_FOLDER}")
    else:
        for file in archivos:
            nombre = os.path.splitext(file)[0]
            filepath = os.path.join(IMAGES_FOLDER, file)
            generar_vector(filepath, nombre)
