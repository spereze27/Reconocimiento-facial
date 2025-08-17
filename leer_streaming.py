import cv2
import os
import pickle
import numpy as np
import mediapipe as mp
from deepface import DeepFace

# --- CONFIGURACIÓN INICIAL ---

# Inicializar MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Modelo para la generación de embeddings. DeepFace soporta varios.
# 'VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib'
# ** LÍNEA CORREGIDA **
MODELO_EMBEDDING = 'Facenet512'

# Ruta a la base de datos de embeddings
DB_PATH = "database"
# Nombre del archivo de la base de datos
DB_FILE = os.path.join(DB_PATH, "representaciones.pkl")

# Umbral de similitud. Si la distancia es menor, se considera una coincidencia.
# Este valor puede necesitar ajuste. Un valor más bajo es más estricto.
UMBRAL_SIMILITUD = 0.60 # Ajusta este valor según tus pruebas

# Crear el directorio de la base de datos si no existe
os.makedirs(DB_PATH, exist_ok=True)


# --- FUNCIONES PRINCIPALES ---

def detectar_y_recortar_rostro(imagen):
    """
    Detecta el rostro en una imagen usando MediaPipe y lo recorta.
    
    Args:
        imagen (np.array): La imagen en formato OpenCV (BGR).
        
    Returns:
        np.array: La imagen del rostro recortado, o None si no se detecta ningún rostro.
    """
    # Convertir la imagen de BGR a RGB, que es el formato que espera MediaPipe
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    
    # Procesar la imagen para detectar rostros
    results = face_detection.process(imagen_rgb)
    
    if not results.detections:
        return None # No se detectaron rostros
        
    # Asumimos que solo hay un rostro de interés (el más grande o el primero)
    detection = results.detections[0]
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = imagen.shape
    
    # Obtener las coordenadas del bounding box y convertirlas a píxeles
    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                 int(bboxC.width * iw), int(bboxC.height * ih)

    # Añadir un pequeño margen para asegurar que se captura todo el rostro
    margen_x = int(w * 0.15)
    margen_y = int(h * 0.15)
    
    x_start = max(0, x - margen_x)
    y_start = max(0, y - margen_y)
    x_end = min(iw, x + w + margen_x)
    y_end = min(ih, y + h + margen_y)
    
    # Recortar el rostro de la imagen original
    rostro_recortado = imagen[y_start:y_end, x_start:x_end]
    
    return rostro_recortado


def registrar_persona_desde_carpeta(ruta_carpeta_imagenes):
    """
    Procesa todas las imágenes en una carpeta, genera sus embeddings y los guarda.
    El nombre de la persona se extrae del nombre del archivo.
    """
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            representaciones = pickle.load(f)
    else:
        representaciones = []

    for nombre_archivo in os.listdir(ruta_carpeta_imagenes):
        ruta_completa = os.path.join(ruta_carpeta_imagenes, nombre_archivo)
        if ruta_completa.lower().endswith(('.png', '.jpg', '.jpeg')):
            
            nombre_persona = os.path.splitext(nombre_archivo)[0].replace("_", " ")
            print(f"Procesando registro para: {nombre_persona}...")
            
            # Verificar si la persona ya está registrada
            if any(item[0] == nombre_persona for item in representaciones):
                print(f"-> {nombre_persona} ya está en la base de datos. Saltando.")
                continue

            imagen = cv2.imread(ruta_completa)
            if imagen is None:
                print(f"  -> Error al leer la imagen: {nombre_archivo}")
                continue
            
            rostro = detectar_y_recortar_rostro(imagen)
            if rostro is None:
                print(f"  -> No se detectó rostro en la imagen de {nombre_persona}.")
                continue
            
            try:
                # Generar el embedding del rostro
                embedding = DeepFace.represent(
                    img_path=rostro,
                    model_name=MODELO_EMBEDDING,
                    enforce_detection=False # Ya hemos hecho la detección
                )[0]["embedding"]
                
                # Guardar el nombre y el embedding
                representaciones.append([nombre_persona, embedding])
                print(f"  -> Registro de {nombre_persona} completado exitosamente.")
                
            except Exception as e:
                print(f"  -> Ocurrió un error al generar el embedding para {nombre_persona}: {e}")

    # Guardar la lista actualizada en el archivo pickle
    with open(DB_FILE, "wb") as f:
        pickle.dump(representaciones, f)
    
    print("\n¡Proceso de registro finalizado!")


def encontrar_identidad(imagen_en_vivo, db):
    """
    Busca la identidad de un rostro en la base de datos de embeddings.
    
    Args:
        imagen_en_vivo (np.array): El frame de la cámara.
        db (list): La lista de representaciones cargada ([nombre, embedding]).
        
    Returns:
        str: El nombre de la persona identificada o "Desconocido".
        float: El valor de la mejor similitud encontrada.
    """
    rostro = detectar_y_recortar_rostro(imagen_en_vivo)
    if rostro is None:
        return "Desconocido", 0.0 # Se retorna "Desconocido" en lugar de None para evitar errores

    try:
        embedding_vivo = DeepFace.represent(
            img_path=rostro,
            model_name=MODELO_EMBEDDING,
            enforce_detection=False
        )[0]["embedding"]
    except Exception as e:
        # print(f"Error al generar embedding en vivo: {e}")
        return "Desconocido", 0.0
        
    identidad_encontrada = "Desconocido"
    mejor_similitud = 0.0
    
    for nombre, embedding_guardado in db:
        # Calcular la similitud del coseno
        embedding_vivo_np = np.asarray(embedding_vivo)
        embedding_guardado_np = np.asarray(embedding_guardado)
        
        similitud = np.dot(embedding_vivo_np, embedding_guardado_np) / \
                    (np.linalg.norm(embedding_vivo_np) * np.linalg.norm(embedding_guardado_np))

        if similitud > mejor_similitud:
            mejor_similitud = similitud
            if similitud > UMBRAL_SIMILITUD:
                identidad_encontrada = nombre
                
    return identidad_encontrada, mejor_similitud


# --- BLOQUE PRINCIPAL DE EJECUCIÓN ---

if __name__ == "__main__":
    # --- FASE 1: REGISTRO DE PERSONAS ---
    # Descomenta la siguiente línea la primera vez que ejecutes el script
    # o cuando añadas nuevas fotos a la carpeta de estudiantes.
    registrar_persona_desde_carpeta("/home/simonperez/GIT/Reconocimiento-facial/data/estudiantes")
    
    # --- FASE 2: RECONOCIMIENTO EN TIEMPO REAL ---
    try:
        with open(DB_FILE, "rb") as f:
            db_representaciones = pickle.load(f)
        if not db_representaciones:
            print("La base de datos está vacía. Por favor, registra algunas personas primero.")
            exit()
        print(f"Base de datos cargada. {len(db_representaciones)} personas registradas.")
    except FileNotFoundError:
        print("Archivo de base de datos no encontrado. Ejecuta la fase de registro primero.")
        exit()

    # Iniciar captura de video
    cap = cv2.VideoCapture(0) # 0 es la cámara por defecto
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        exit()
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Realizar la identificación en el frame actual
        identidad, similitud = encontrar_identidad(frame, db_representaciones)
        
        texto_display = f"Persona: {identidad}"
        color_texto = (0, 255, 0) if identidad != "Desconocido" else (0, 0, 255)
        
        # Mostrar el resultado en la pantalla
        cv2.putText(frame, texto_display, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_texto, 2, cv2.LINE_AA)
        if identidad != "Desconocido":
            cv2.putText(frame, f"Similitud: {similitud:.2f}", (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_texto, 2, cv2.LINE_AA)
        
        cv2.imshow('Reconocimiento Facial en Vivo', frame)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()