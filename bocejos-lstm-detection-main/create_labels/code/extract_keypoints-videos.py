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
ANNOTATION_SUFFIX = ".txt"  # nome do txt deve bater com nome do vídeo, ex: video1.mp4 -> video1_labels.txt
FRAME_STEP = 1  # processa a cada frame

# ========== FACE MESH ==========
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, refine_landmarks=True)

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
    """Carrega anotações txt do tipo '00:49:00 - 00:53:00 --> 1'"""
    ranges = []
    with open(label_path, "r") as f:
        for line in f:
            match = re.match(r"(\d+:\d+:\d+)\s*-\s*(\d+:\d+:\d+)\s*-->\s*(\d+)", line.strip())
            if match:
                start, end, label = match.groups()
                ranges.append((parse_time(start), parse_time(end), int(label)))
    return ranges

def get_label_for_time(t, label_ranges):
    """Retorna o label correspondente ao segundo `t`"""
    for start, end, label in label_ranges:
        if start <= t <= end:
            return label
    return 0  # default para "alerta"









def euclidean(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

left_eye_idx = 33
right_eye_idx = 263






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
    print(label_ranges)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % FRAME_STEP != 0:
            frame_number += 1
            continue

        timestamp_sec = frame_number / fps
        
        if label_ranges:
            label = get_label_for_time(timestamp_sec, label_ranges)
        else:
            label = 0
        
        # if label != 0:
        #     print(f"Frame: {frame_number}, Time: {timestamp_sec:.2f}, Label: {label}")
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        frame_record = {
            "video_id": os.path.splitext(video_file)[0],
            "frame_number": frame_number,
            "driver_state": label,
            "keypoints": {}
        }
        
        
        #EUCLIDEAN DISTANCE
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            cx, cy = landmarks[1].x, landmarks[1].y
        
            # Distância usada para normalizar (largura dos olhos)
            if left_eye_idx < len(landmarks) and right_eye_idx < len(landmarks):
                scale = euclidean(landmarks[left_eye_idx], landmarks[right_eye_idx])
            else:
                scale = 1.0
        
            if scale < 1e-5:  # evita divisão por zero
                scale = 1.0
        
            for idx in all_landmark_indices:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    x = (lm.x - cx) / scale
                    y = (lm.y - cy) / scale
                    frame_record["keypoints"][str(idx)] = [x, y]
                else:
                    frame_record["keypoints"][str(idx)] = [0.0, 0.0]
                    
        # if results.multi_face_landmarks:
        #     landmarks = results.multi_face_landmarks[0].landmark
            
        #     #NORMAL
        #     # for idx in all_landmark_indices:
        #     #     if idx < len(landmarks):
        #     #         lm = landmarks[idx]
        #     #         # x = lm.x * frame.shape[1]
        #     #         # y = lm.y * frame.shape[0]
                    
        #     #         x = lm.x
        #     #         y = lm.y
        #     #         frame_record["keypoints"][str(idx)] = [x, y]
            
        #     #NOSE
        #     nose = landmarks[1]
        #     cx, cy = nose.x, nose.y
            
        #     for idx in all_landmark_indices:
        #         if idx < len(landmarks):
        #             lm = landmarks[idx]
        #             x = lm.x - cx
        #             y = lm.y - cy
        #             frame_record["keypoints"][str(idx)] = [x, y]
        #         else:
        #             frame_record["keypoints"][str(idx)] = [0.0, 0.0]
        else:
            for idx in all_landmark_indices:
                frame_record["keypoints"][str(idx)] = [0.0, 0.0]

        output_data.append(frame_record)
        frame_number += 1

    cap.release()

    # salvar json
    output_json = os.path.join(OUTPUT_DIR, f"{os.path.splitext(video_file)[0]}_keypoints_1_euclidean_normalized2.json")
    with open(output_json, "w") as f:
        json.dump(output_data, f)

    print(f"[✓] JSON salvo em {output_json}")

