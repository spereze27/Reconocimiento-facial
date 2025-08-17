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

# Modelo para la generación de embeddings
MODELO_EMBEDDING = 'Facenet512'

# Ruta a la base de datos
DB_PATH = "database"
DB_FILE = os.path.join(DB_PATH, "representaciones.pkl")

# Umbral de similitud
UMBRAL_SIMILITUD = 0.5

# Crear el directorio si no existe
os.makedirs(DB_PATH, exist_ok=True)


# --- FUNCIONES PRINCIPALES ---

def detectar_rostros(imagen):
    """
    Detecta TODOS los rostros en una imagen y devuelve sus recortes y coordenadas.
    
    Args:
        imagen (np.array): La imagen en formato OpenCV (BGR).
        
    Returns:
        list: Una lista de diccionarios. Cada diccionario contiene:
              'rostro': la imagen del rostro recortado.
              'box': las coordenadas (x, y, w, h) del rectángulo.
    """
    lista_rostros = []
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    results = face_detection.process(imagen_rgb)
    
    if not results.detections:
        return lista_rostros # Devuelve lista vacía si no hay rostros
        
    ih, iw, _ = imagen.shape
    
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                     int(bboxC.width * iw), int(bboxC.height * ih)

        # Pequeño ajuste para evitar coordenadas negativas
        x, y = max(0, x), max(0, y)

        rostro_recortado = imagen[y:y+h, x:x+w]
        
        # Asegurarse que el recorte no esté vacío
        if rostro_recortado.size > 0:
            lista_rostros.append({
                'rostro': rostro_recortado,
                'box': (x, y, w, h)
            })
            
    return lista_rostros


def registrar_persona_desde_carpeta(ruta_carpeta_imagenes):
    """
    Procesa todas las imágenes en una carpeta, genera sus embeddings y los guarda.
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
            
            if any(item[0] == nombre_persona for item in representaciones):
                print(f"-> {nombre_persona} ya está en la base de datos. Saltando.")
                continue

            imagen = cv2.imread(ruta_completa)
            if imagen is None:
                print(f"  -> Error al leer la imagen: {nombre_archivo}")
                continue
            
            # Usamos la nueva función que detecta múltiples rostros, pero solo usamos el primero para el registro
            rostros_detectados = detectar_rostros(imagen)
            if not rostros_detectados:
                print(f"  -> No se detectó rostro en la imagen de {nombre_persona}.")
                continue
            
            # Usamos el primer rostro detectado para el registro
            rostro = rostros_detectados[0]['rostro']
            
            try:
                embedding = DeepFace.represent(
                    img_path=rostro,
                    model_name=MODELO_EMBEDDING,
                    enforce_detection=False
                )[0]["embedding"]
                
                representaciones.append([nombre_persona, embedding])
                print(f"  -> Registro de {nombre_persona} completado exitosamente.")
                
            except Exception as e:
                print(f"  -> Ocurrió un error al generar el embedding para {nombre_persona}: {e}")

    with open(DB_FILE, "wb") as f:
        pickle.dump(representaciones, f)
    
    print("\n¡Proceso de registro finalizado!")


def encontrar_identidad(rostro_recortado, db):
    """
    Busca la identidad de un rostro ya recortado en la base de datos.
    
    Args:
        rostro_recortado (np.array): La imagen del rostro ya recortado.
        db (list): La lista de representaciones cargada ([nombre, embedding]).
        
    Returns:
        str: El nombre de la persona o "Desconocido".
        float: El valor de la mejor similitud.
    """
    try:
        embedding_vivo = DeepFace.represent(
            img_path=rostro_recortado,
            model_name=MODELO_EMBEDDING,
            enforce_detection=False
        )[0]["embedding"]
    except Exception:
        return "Error", 0.0
        
    identidad_encontrada = "Desconocido"
    mejor_similitud = 0.0
    
    for nombre, embedding_guardado in db:
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
    # --- FASE 1: REGISTRO (si es necesario) ---
    # registrar_persona_desde_carpeta("ruta/a/tu/carpeta/de/registro")
    
    # --- FASE 2: RECONOCIMIENTO DESDE CARPETA ---
    try:
        with open(DB_FILE, "rb") as f:
            db_representaciones = pickle.load(f)
        if not db_representaciones:
            print("La base de datos está vacía. Registra algunas personas primero.")
            exit()
        print(f"✅ Base de datos cargada con {len(db_representaciones)} personas.")
    except FileNotFoundError:
        print("❌ Archivo de base de datos no encontrado. Ejecuta el registro primero.")
        exit()

    # ** CAMBIA ESTA RUTA a la carpeta con las fotos que quieres verificar **
    carpeta_a_verificar = "/home/simonperez/GIT/Reconocimiento-facial/fotos_test" 

    if not os.path.isdir(carpeta_a_verificar):
        print(f"❌ Error: La carpeta '{carpeta_a_verificar}' no existe.")
        exit()

    for nombre_archivo in os.listdir(carpeta_a_verificar):
        ruta_imagen = os.path.join(carpeta_a_verificar, nombre_archivo)
        
        if ruta_imagen.lower().endswith(('.png', '.jpg', '.jpeg')):
            frame = cv2.imread(ruta_imagen)
            if frame is None:
                print(f"No se pudo leer la imagen: {nombre_archivo}")
                continue

            # Detectar todos los rostros en la imagen
            rostros_encontrados = detectar_rostros(frame)
            
            # Si no se encontraron rostros, simplemente mostrar la imagen original
            if not rostros_encontrados:
                print(f"ℹ️ No se detectaron rostros en {nombre_archivo}")
            
            # Iterar sobre cada rostro encontrado
            for info_rostro in rostros_encontrados:
                rostro_img = info_rostro['rostro']
                (x, y, w, h) = info_rostro['box']

                # Encontrar la identidad para este rostro específico
                identidad, similitud = encontrar_identidad(rostro_img, db_representaciones)
                
                color = (0, 255, 0) if identidad != "Desconocido" else (0, 0, 255)
                texto_display = f"{identidad} ({similitud:.2f})"
                
                # --- DIBUJAR EL RECTÁNGULO Y EL TEXTO ---
                # 1. Dibujar el rectángulo alrededor del rostro
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # 2. Dibujar el fondo para el texto
                cv2.rectangle(frame, (x, y - 30), (x + w, y), color, -1)
                
                # 3. Poner el texto con la identidad
                cv2.putText(frame, texto_display, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Mostrar la imagen final con todos los rostros marcados
            cv2.imshow(f'Reconocimiento - {nombre_archivo}', frame)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            else:
                cv2.destroyWindow(f'Reconocimiento - {nombre_archivo}')
    
    cv2.destroyAllWindows()