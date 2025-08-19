# main_streaming.py

import cv2
import asyncio
import mediapipe as mp
from deteccion_facial import DeteccionFacial # Asegúrate de que tu clase esté en un archivo llamado deteccion_facial.py

async def main():
    """
    Función principal para capturar video de la cámara, procesar los gestos faciales
    y mostrar los resultados en tiempo real.
    """
    # 1. Inicialización
    cap = cv2.VideoCapture(0) # Inicia la captura de video (0 es la cámara por defecto)
    detector = DeteccionFacial() # Crea una instancia de tu clase

    # Inicializa el modelo de MediaPipe Face Mesh una sola vez para mayor eficiencia
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    if not cap.isOpened():
        print("Error: No se puede abrir la cámara.")
        return

    # 2. Bucle principal de procesamiento
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No se pudo recibir el fotograma. Saliendo...")
            break

        # Invierte el fotograma horizontalmente para un efecto espejo
        frame = cv2.flip(frame, 1)

        # 3. Llamada asíncrona a la detección de gestos
        try:
            gestures = await detector.detect_facial_gestures(frame, face_mesh)
            mouth_open, looking_center, eyes_closed, face_detected = gestures
            
            # 4. Muestra los resultados en el fotograma
            if face_detected:
                color_ojos = (0, 255, 0) if not eyes_closed else (0, 0, 255)
                texto_ojos = "Ojos: Abiertos" if not eyes_closed else "Ojos: Cerrados"

                color_boca = (0, 0, 255) if mouth_open else (0, 255, 0)
                texto_boca = "Boca: Abierta" if mouth_open else "Boca: Cerrada"
                
                color_mirada = (0, 255, 0) if looking_center else (0, 0, 255)
                texto_mirada = "Mirada: Al frente" if looking_center else "Mirada: A los lados"
                
                cv2.putText(frame, texto_ojos, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_ojos, 2)
                cv2.putText(frame, texto_boca, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color_boca, 2)
                cv2.putText(frame, texto_mirada, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color_mirada, 2)

        except Exception as e:
            # En caso de que no se detecte una cara, fr.face_encodings puede lanzar un error
            # o si ocurre otro problema durante el procesamiento.
            cv2.putText(frame, "No se detecta rostro", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        # Muestra el fotograma resultante
        cv2.imshow('Deteccion de Gestos Faciales', frame)

        # 5. Condición de salida
        # Presiona 'q' para salir del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. Liberación de recursos
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    asyncio.run(main())
