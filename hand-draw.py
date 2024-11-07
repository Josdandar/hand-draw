import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Inicializar MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Variables para almacenar la posición anterior
prev_x, prev_y = None, None

def main():
    global prev_x, prev_y
    cap = cv2.VideoCapture(0)  # Iniciar la cámara

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        cap.release()
        return

    # Crear la imagen de dibujo con el mismo tamaño y tipo que el frame
    drawing = np.zeros_like(frame)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Preparar el frame
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Obtener la punta del índice y el pulgar
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # Coordenadas de la punta del índice para dibujar
                ix, iy = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
                tx, ty = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])

                # Distancia entre índice y pulgar para decidir si dibujar o borrar
                distance = np.sqrt((ix - tx) ** 2 + (iy - ty) ** 2)

                if distance < 40:  # Si los dedos están cerca, dibujar
                    if prev_x is not None and prev_y is not None:
                        # Dibujar una línea desde la posición anterior a la actual
                        cv2.line(drawing, (prev_x, prev_y), (ix, iy), (0, 255, 0), 5)
                    # Actualizar la posición anterior
                    prev_x, prev_y = ix, iy
                elif distance > 60:  # Si los dedos están lejos, borrar
                    # Dibujar un círculo negro en la posición del dedo índice para borrar
                    cv2.circle(drawing, (ix, iy), 20, (0, 0, 0), -1)
                    # Reiniciar la posición anterior para evitar líneas continuas al volver a dibujar
                    prev_x, prev_y = None, None

        # Combinar frame y dibujo
        combined = cv2.addWeighted(frame, 0.5, drawing, 0.5, 0)

        cv2.imshow('Hand Tracking', combined)

        # Opciones de teclado
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC para salir
            break
        elif key == ord('c'):  # 'c' para limpiar toda la pantalla
            drawing = np.zeros_like(frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
