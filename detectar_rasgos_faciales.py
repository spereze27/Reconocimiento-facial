# deteccion_facial.py (versión revisada)

import cv2
import mediapipe as mp
import face_recognition as fr
import math
import numpy as np
import concurrent.futures
import asyncio

class DeteccionFacial:
    def __init__(self):
        # Inicializa las configuraciones y modelos necesarios aquí
        self._EAR_THRESH = 0.2
        self._index_left_eye = [33, 160, 158, 133, 153, 144]
        self._index_right_eye = [362, 385, 387, 263, 373, 380]
        self._L_EYE_L = [33]  # Extremo izquierdo del ojo izquierdo (usamos 33 en vez de 263 que es del ojo derecho)
        self._L_EYE_R = [133] # Extremo derecho del ojo izquierdo
        self._L_DOWN = [145]  # Punto inferior del ojo izquierdo
        self._L_UP = [159]    # Punto superior del ojo izquierdo
        self._LEFT_IRIS = [474, 475, 476, 477]
        self._RIGHT_IRIS = [469, 470, 471, 472]
        
        # Puntos clave para el MAR (Mouth Aspect Ratio)
        self._LIPS_UP = 13
        self._LIPS_DOWN = 14
        self._LIPS_LEFT = 61
        self._LIPS_RIGHT = 291

    async def vectorized_face(self, frame):
        def _vectorized_face():
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = fr.face_encodings(image)
            # Asegurarse de que se encontró al menos una cara
            return encodings[0] if encodings else None
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await asyncio.get_event_loop().run_in_executor(executor, _vectorized_face)
    
    def _distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    
    # Función mejorada para detectar si la boca está abierta usando MAR
    def _is_mouth_open(self, mesh_points):
        # Calcula la distancia vertical (altura) y horizontal (anchura) de la boca
        mouth_height = self._distance(mesh_points[self._LIPS_UP], mesh_points[self._LIPS_DOWN])
        mouth_width = self._distance(mesh_points[self._LIPS_LEFT], mesh_points[self._LIPS_RIGHT])
        
        # Evitar división por cero
        if mouth_width == 0:
            return False
            
        mar = mouth_height / mouth_width
        threshold = 0.5  # Este umbral puede necesitar ajuste
        return mar > threshold

    def _is_looking_center(self, iris_center, eye_right_point, eye_left_point):
        center_to_right_dist = self._distance(iris_center, eye_right_point)
        total_dist = self._distance(eye_right_point, eye_left_point)
        
        if total_dist == 0: return True # Evitar división por cero

        ratio = center_to_right_dist / total_dist
        # Si el ratio está aproximadamente en el centro (ej. entre 35% y 65%)
        return 0.35 < ratio < 0.65

    def _is_looking_down(self, iris_center, eye_up_point, eye_down_point):
        center_to_down_dist = self._distance(iris_center, eye_down_point)
        total_dist = self._distance(eye_up_point, eye_down_point)

        if total_dist == 0: return False # Evitar división por cero

        ratio = center_to_down_dist / total_dist
        # Si el iris está en la mitad inferior del ojo
        return ratio < 0.5
    
    def _eye_aspect_ratio(self, coordinates):
        p2_p6 = self._distance(coordinates[1], coordinates[5])
        p3_p5 = self._distance(coordinates[2], coordinates[4])
        p1_p4 = self._distance(coordinates[0], coordinates[3])
        
        if p1_p4 == 0: return 0.0 # Evitar división por cero
            
        return (p2_p6 + p3_p5) / (2.0 * p1_p4)

    def _get_coordinates(self, landmarks, indices, width, height):
        return [
            [int(landmarks[i].x * width), int(landmarks[i].y * height)]
            for i in indices
        ]

    async def detect_facial_gestures(self, frame, face_mesh):
        def _detect_facial_gestures():
            height, width, _ = frame.shape
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if not results.multi_face_landmarks:
                # Retorno consistente cuando no hay cara
                return [False, False, False, False] # [mouth_open, looking_center, eyes_closed, face_detected]
            
            # Asumimos una sola cara para simplificar
            face_landmarks = results.multi_face_landmarks[0]
            mesh_points = np.array([
                [int(p.x * width), int(p.y * height)] for p in face_landmarks.landmark
            ])

            # 1. Detección de boca abierta
            mouth_open_status = self._is_mouth_open(mesh_points)
            
            # 2. Detección de dirección de la mirada (usando ojo izquierdo)
            (cx, cy), _ = cv2.minEnclosingCircle(mesh_points[self._LEFT_IRIS])
            iris_center = np.array([cx, cy], dtype=np.int32)
            
            looking_center_status = self._is_looking_center(iris_center, mesh_points[self._L_EYE_R][0], mesh_points[self._L_EYE_L][0])
            # looking_down_status = self._is_looking_down(iris_center, mesh_points[self._L_UP][0], mesh_points[self._L_DOWN][0]) # Descomentar si se necesita

            # 3. Detección de ojos cerrados (EAR)
            coords_left_eye = self._get_coordinates(face_landmarks, self._index_left_eye, width, height)
            coords_right_eye = self._get_coordinates(face_landmarks, self._index_right_eye, width, height)
            
            ear_left = self._eye_aspect_ratio(coords_left_eye)
            ear_right = self._eye_aspect_ratio(coords_right_eye)
            
            avg_ear = (ear_left + ear_right) / 2.0
            eyes_closed_status = avg_ear < self._EAR_THRESH
            
            return [mouth_open_status, looking_center_status, eyes_closed_status, True]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await asyncio.get_event_loop().run_in_executor(executor, _detect_facial_gestures)
