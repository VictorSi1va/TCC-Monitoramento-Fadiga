import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import onnxruntime as ort

# ==== Parâmetros ====
WINDOW_SIZE = 45
MODEL_PATH = "./create_labels/code/documentos/model_lstm_3_45_euclidean_hands.onnx"

# ==== Inicializar MediaPipe ====
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# ==== Índices dos keypoints usados ====
FACE_IDX = {
    "mouth_outer":   [61, 40, 37, 0, 267, 270, 291],  # boca externa (pontos principais)
    "mouth_inner":   [78, 95, 14, 317, 308],          # boca interna (centro e extremos)
    "left_eye":      [33, 160, 158, 133, 153, 144, 145],  # contorno da pálpebra esquerda
    "right_eye":     [362, 385, 387, 263, 373, 380, 381], # contorno da pálpebra direita
    "left_brow":     [70, 105, 107],                   # 3 pontos-chave
    "right_brow":    [336, 334, 300],
    "nose":          [1, 2, 168],                      # ponta + centro
    "chin":          [152]                             # queixo central
}
all_landmark_indices = sorted(set(idx for group in FACE_IDX.values() for idx in group))

# ==== ONNX Runtime ====
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# ==== Janela de features ====
window = deque(maxlen=WINDOW_SIZE)

# ==== Webcam ====
cap = cv2.VideoCapture(0)

# ==== Função auxiliar ====
def euclidean(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2) ** 0.5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    keypoints = []

    # ==== FACE ====
    if face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark

        # Centro: nariz (1)
        cx, cy = landmarks[1].x, landmarks[1].y

        # Escala: distância entre os olhos
        scale = euclidean(landmarks[33], landmarks[263])
        if scale < 1e-5:
            scale = 1.0

        for idx in all_landmark_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                x = (lm.x - cx) / scale
                y = (lm.y - cy) / scale
                keypoints.extend([x, y])

                # Visualização
                px = int(lm.x * image.shape[1])
                py = int(lm.y * image.shape[0])
                cv2.circle(image, (px, py), 2, (0, 255, 0), -1)
            else:
                keypoints.extend([0.0, 0.0])
    else:
        keypoints.extend([0.0, 0.0] * len(all_landmark_indices))

    # ==== HANDS ====
    hands_data = {"left": {}, "right": {}}
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            label = handedness.classification[0].label.lower()  # 'left' or 'right'
            landmarks = hand_landmarks.landmark

            # Centro: punho
            cx, cy = landmarks[0].x, landmarks[0].y

            # Escala: punho até ponta do dedo médio
            scale = euclidean(landmarks[0], landmarks[12])
            if scale < 1e-5:
                scale = 1.0

            for idx, lm in enumerate(landmarks):
                x = (lm.x - cx) / scale
                y = (lm.y - cy) / scale
                hands_data[label][idx] = (x, y)

                # Visualização
                px = int(lm.x * image.shape[1])
                py = int(lm.y * image.shape[0])
                color = (255, 0, 0) if label == "left" else (0, 0, 255)
                cv2.circle(image, (px, py), 3, color, -1)

    # Adiciona keypoints normalizados das mãos
    for side in ["left", "right"]:
        for i in range(21):
            if i in hands_data[side]:
                keypoints.extend(list(hands_data[side][i]))
            else:
                keypoints.extend([0.0, 0.0])

    # ==== JANELA ====
    window.append(keypoints)

    if len(window) == WINDOW_SIZE:
        x_input = np.array([window], dtype=np.float32)
        prediction = session.run(None, {input_name: x_input})[0][0]
        predicted_class = int(np.argmax(prediction)) if prediction.shape[-1] > 1 else int(prediction > 0.5)

        label_str = ["Alerta", "Bocejando", "Microsleep"][predicted_class]
        if predicted_class != 0:
            print(predicted_class)
    else:
        label_str = "Carregando..."

    # ==== VISUAL ====
    cv2.putText(image, f"Estado: {label_str}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)

    cv2.imshow("Webcam - Detecção de Fadiga (ONNX)", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
