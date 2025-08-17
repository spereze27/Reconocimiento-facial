import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace

# ===============================
# --- CONFIGURACIÓN ---
# ===============================

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)

# Modelo de DeepFace para generar embeddings
MODELO_EMBEDDING = 'Facenet512'

# ===============================
# --- FUNCIONES ---
# ===============================

def detectar_y_recortar_rostro(imagen):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    results = face_detection.process(imagen_rgb)
    
    if not results.detections:
        return None
    
    detection = results.detections[0]
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = imagen.shape
    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                 int(bboxC.width * iw), int(bboxC.height * ih)
    
    margen_x = int(w * 0.15)
    margen_y = int(h * 0.15)
    
    x_start = max(0, x - margen_x)
    y_start = max(0, y - margen_y)
    x_end = min(iw, x + w + margen_x)
    y_end = min(ih, y + h + margen_y)
    
    return imagen[y_start:y_end, x_start:x_end]

def dibujar_embedding_en_imagen(frame, embedding, x=20, y=80, line_height=20, valores_por_linea=8):
    """
    Dibuja el embedding como texto en la imagen, dividiéndolo en varias líneas.
    """
    for i in range(0, len(embedding), valores_por_linea):
        fragmento = embedding[i:i+valores_por_linea]
        texto_linea = ", ".join([f"{v:.2f}" for v in fragmento])
        cv2.putText(frame, texto_linea, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
        y += line_height

# ===============================
# --- EJECUCIÓN EN VIVO ---
# ===============================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # --- Mallado facial ---
    results_mesh = face_mesh.process(frame_rgb)
    if results_mesh.multi_face_landmarks:
        for face_landmarks in results_mesh.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )
    
    # --- Embedding ---
    rostro = detectar_y_recortar_rostro(frame)
    if rostro is not None:
        try:
            rostro_rgb = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)
            embedding = DeepFace.represent(
                img_path=rostro_rgb,
                model_name=MODELO_EMBEDDING,
                enforce_detection=False
            )[0]["embedding"]
            
            # Dibujar embedding completo en la imagen
            dibujar_embedding_en_imagen(frame, embedding, x=20, y=80, line_height=20, valores_por_linea=8)
            
        except Exception as e:
            cv2.putText(frame, "Error al generar embedding", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("Mallado Facial + Embedding en Imagen", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
