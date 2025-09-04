import cv2
import mediapipe as mp
import json
import time

# Inicializar MediaPipe
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Criar instâncias dos modelos
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    # refine_landmarks=True  # ATIVA os pontos da íris
)

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Webcam
cap = cv2.VideoCapture(0)

# Keypoints relevantes (índices do FaceMesh)
# FACE_IDX = {
#     "mouth_outer":   [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
#     "mouth_inner":   [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
#     "left_eye":      [33, 160, 158, 133, 153, 144, 145],  # contorno da pálpebra esquerda
#     "right_eye":     [362, 385, 387, 263, 373, 380, 381], # contorno da pálpebra direita
#     "left_brow":     [70, 63, 105, 66, 107],              # sobrancelha esquerda
#     "right_brow":    [336, 296, 334, 293, 300],           # sobrancelha direita
#     "nose":          [1, 2, 98, 327, 168],
#     "chin":          [152]
# }

# FACE_IDX = {
#     "mouth_outer":   [61, 40, 37, 0, 267, 270, 291],  # boca externa (pontos principais)
#     "mouth_inner":   [78, 95, 14, 317, 308],          # boca interna (centro e extremos)
#     "left_eye":      [33, 160, 158, 133, 153, 144, 145],  # contorno da pálpebra esquerda
#     "right_eye":     [362, 385, 387, 263, 373, 380, 381], # contorno da pálpebra direita
#     "left_brow":     [70, 105, 107],                   # 3 pontos-chave
#     "right_brow":    [336, 334, 300],
#     "nose":          [1, 2, 168],                      # ponta + centro
#     "chin":          [152]                             # queixo central
# }

FACE_IDX = {
    # "mouth_outer":   [61, 40, 37, 0, 267, 270, 291],  # boca externa (pontos principais)
    # "mouth_inner":   [78, 95, 14, 317, 308],          # boca interna (centro e extremos)
    "left_eye":      [33],  # contorno da pálpebra esquerda
    "right_eye":     [263], # contorno da pálpebra direita
    # "left_brow":     [70, 105, 107],                   # 3 pontos-chave
    # "right_brow":    [336, 334, 300],
    # "nose":          [1, 2, 168],                      # ponta + centro
    # "chin":          [152]                             # queixo central
}


def extract_face_keypoints(landmarks, w, h):
    keypoints = {}
    for region, indices in FACE_IDX.items():
        keypoints[region] = []
        for idx in indices:
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            keypoints[region].append([x, y])
    return keypoints

def extract_hand_keypoints(landmarks, w, h):
    keypoints = []
    for lm in landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        keypoints.append([x, y])
    return keypoints

# Para salvar os dados por frame
frame_data = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    # Processar frame
    face_results = face_mesh.process(rgb)
    hands_results = hands.process(rgb)

    keypoints_frame = {}

    # Rosto
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # mp_drawing.draw_landmarks(
            #     frame, face_landmarks, mp_face.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            # )
            # face_kps = extract_face_keypoints(face_landmarks.landmark, w, h)
            # keypoints_frame["face"] = face_kps

            # # Mostrar pontos na tela
            # for region in face_kps:
            #     for x, y in face_kps[region]:
            #         cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            face_kps = extract_face_keypoints(face_landmarks.landmark, w, h)
            keypoints_frame["face"] = face_kps
        
            for region in face_kps:
                for x, y in face_kps[region]:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
    # Mãos
    hands_kps = {"left": None, "right": None}
    if hands_results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            label = hands_results.multi_handedness[i].classification[0].label.lower()
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hands_kps[label] = extract_hand_keypoints(hand_landmarks, w, h)

    keypoints_frame["hands"] = hands_kps

    # Salvar frame atual
    frame_id = f"frame_{int(time.time() * 1000)}"
    frame_data[frame_id] = keypoints_frame

    cv2.imshow("Extract Keypoints", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()

# # Salvar em JSON
# with open("keypoints_output.json", "w") as f:
#     json.dump(frame_data, f, indent=2)

# print("Keypoints salvos em keypoints_output.json")
