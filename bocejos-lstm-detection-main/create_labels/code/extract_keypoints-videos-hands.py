import os
import cv2
import json
import re
from tqdm import tqdm
import mediapipe as mp
from datetime import timedelta

# ========== CONFIGURAÇÕES ==========
VIDEO_DIR = "../datasets/Videos Fadiga"
OUTPUT_DIR = "../jsons"
ANNOTATION_SUFFIX = ".txt"
FRAME_STEP = 1

# ========== MEDIA PIPE ==========
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    refine_landmarks=True
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# ========== FACE LANDMARKS ==========
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
all_landmark_indices = sorted({idx for group in FACE_IDX.values() for idx in group})

# ========== FUNÇÕES AUXILIARES ==========

def parse_time(tstr):
    m, s, ms = map(int, tstr.split(":"))
    return m * 60 + s + ms / 1000

def load_labels(label_path):
    ranges = []
    with open(label_path, "r") as f:
        for line in f:
            match = re.match(r"(\d+:\d+:\d+)\s*-\s*(\d+:\d+:\d+)\s*-->\s*(\d+)", line.strip())
            if match:
                start, end, label = match.groups()
                ranges.append((parse_time(start), parse_time(end), int(label)))
    return ranges

def get_label_for_time(t, label_ranges):
    for start, end, label in label_ranges:
        if start <= t <= end:
            return label
    return 0

def euclidean(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2) ** 0.5

# ========== PROCESSAMENTO ==========

os.makedirs(OUTPUT_DIR, exist_ok=True)
video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

for video_file in video_files:
    video_path = os.path.join(VIDEO_DIR, video_file)
    label_path = os.path.join(VIDEO_DIR, os.path.splitext(video_file)[0] + ANNOTATION_SUFFIX)
    if not os.path.exists(label_path):
        print(f"[!] Anotações não encontradas para {video_file}, pulando.")
        continue

    print(f"[✓] Processando {video_file}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0
    output_data = []
    label_ranges = load_labels(label_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % FRAME_STEP != 0:
            frame_number += 1
            continue

        timestamp_sec = frame_number / fps
        label = get_label_for_time(timestamp_sec, label_ranges)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)
        hand_results = hands.process(rgb)

        frame_record = {
            "video_id": os.path.splitext(video_file)[0],
            "frame_number": frame_number,
            "driver_state": label,
            "keypoints": {},
            "hands": {
                "left": {},
                "right": {}
            }
        }

        # ========== FACE KEYPOINTS (normalizados) ==========
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark

            # Centro: nariz
            cx, cy = landmarks[1].x, landmarks[1].y

            # Escala: distância entre os olhos
            left_eye_idx = 33
            right_eye_idx = 263
            if left_eye_idx < len(landmarks) and right_eye_idx < len(landmarks):
                scale = euclidean(landmarks[left_eye_idx], landmarks[right_eye_idx])
            else:
                scale = 1.0

            if scale < 1e-5:
                scale = 1.0

            for idx in all_landmark_indices:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    x = (lm.x - cx) / scale
                    y = (lm.y - cy) / scale
                    frame_record["keypoints"][str(idx)] = [x, y]
                else:
                    frame_record["keypoints"][str(idx)] = [0.0, 0.0]
        else:
            for idx in all_landmark_indices:
                frame_record["keypoints"][str(idx)] = [0.0, 0.0]

        # ========== HAND KEYPOINTS (normalizados) ==========
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                label_hand = handedness.classification[0].label.lower()  # 'left' or 'right'
                landmarks = hand_landmarks.landmark

                # Centro: punho (landmark 0)
                cx, cy = landmarks[0].x, landmarks[0].y

                # Escala: punho até ponta do dedo médio (landmark 12)
                scale = euclidean(landmarks[0], landmarks[12])
                if scale < 1e-5:
                    scale = 1.0

                for idx, lm in enumerate(landmarks):
                    x = (lm.x - cx) / scale
                    y = (lm.y - cy) / scale
                    frame_record["hands"][label_hand][str(idx)] = [x, y]
        else:
            for side in ["left", "right"]:
                for idx in range(21):
                    frame_record["hands"][side][str(idx)] = [0.0, 0.0]

        output_data.append(frame_record)
        frame_number += 1

    cap.release()

    output_json = os.path.join(OUTPUT_DIR, f"{os.path.splitext(video_file)[0]}_keypoints_1_hands_euclidean_normalized.json")
    with open(output_json, "w") as f:
        json.dump(output_data, f)

    print(f"[✓] JSON salvo em {output_json}")
