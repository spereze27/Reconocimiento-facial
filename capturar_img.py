import cv2
import os

# Ruta base donde se guardarán las fotos
BASE_DIR = "data/estudiantes"

# Crear carpeta base si no existe
os.makedirs(BASE_DIR, exist_ok=True)

# Iniciar captura de video (0 = cámara por defecto)
cap = cv2.VideoCapture(0)

print("Presiona 'p' para capturar la foto o 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Mostrar la cámara en ventana
    cv2.imshow("Captura de estudiante", frame)

    key = cv2.waitKey(1) & 0xFF

    # Si se presiona "p" se guarda la imagen
    if key == ord("p"):
        # Pedir nombre del estudiante
        nombre = input("👉 Ingresa el nombre del estudiante: ").strip()

        if nombre:
            # Crear path final
            filename = f"{nombre}.png"
            filepath = os.path.join(BASE_DIR, filename)

            # Guardar la imagen en disco
            cv2.imwrite(filepath, frame)
            print(f"✅ Imagen guardada en: {filepath}")
        else:
            print("⚠️ Nombre vacío, no se guardó la imagen.")

    # Salir con "q"
    elif key == ord("q"):
        print("Saliendo...")
        break

cap.release()
cv2.destroyAllWindows()
