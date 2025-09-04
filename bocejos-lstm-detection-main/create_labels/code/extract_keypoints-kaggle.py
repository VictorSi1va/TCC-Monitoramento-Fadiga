import json
import os
import cv2
import mediapipe as mp
from tqdm import tqdm
import re

FRAME_STEP = 1

# Inicializar MediaPipe
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    refine_landmarks=True
)


# FACE_IDX com apenas os pontos relevantes (removendo íris)
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

# Flatten all landmark indices into a single list
all_landmark_indices = []
for group in FACE_IDX.values():
    all_landmark_indices.extend(group)
all_landmark_indices = sorted(list(set(all_landmark_indices)))  # Remove duplicates


# Mapeamento de labels
label_map = {
    "alert": 0,
    "yawning": 1,
    "microsleep": 2
}


# Caminho do JSON
annotation_path = "../datasets/FL3D - Kaggle/annotations_all.json"
with open(annotation_path, "r") as f:
    annotations = json.load(f)

output_data = []


frame_number = 0




def euclidean(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

left_eye_idx = 33
right_eye_idx = 263




for i, (rel_path, data) in enumerate(tqdm(annotations.items(), desc="Processando frames")):
# for rel_path, data in list(annotations.items())[0:500]:
    
    if frame_number % FRAME_STEP != 0:
        frame_number += 1
        continue
    
    abs_path = os.path.join("../datasets/FL3D - Kaggle", rel_path.replace("./classification_frames/", ""))
    
    if not os.path.exists(abs_path):
        continue  # Skip missing files
        
    image = cv2.imread(abs_path)
    if image is None:
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    match = re.search(r"frame(\d+)", rel_path)
    frame_number = int(match.group(1)) if match else -1
    
    video_id = rel_path.split("/")[2]
    
    frame_record = {
        "video_id": video_id,
        "frame_number": frame_number,
        "driver_state": label_map.get(data["driver_state"], -1),
        "keypoints": {}
    }


    # if results.multi_face_landmarks:
    #     landmarks = results.multi_face_landmarks[0].landmark
    #     h, w, _ = image.shape
    
    #     for idx in all_landmark_indices:
    #         if idx < len(landmarks):
    #             lm = landmarks[idx]
    #             x = int(lm.x * w)
    #             y = int(lm.y * h)
    #             cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    #     cv2.imshow("Keypoints", image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    
    
    

    
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
    #     #         # x = lm.x * image.shape[1]
    #     #         # y = lm.y * image.shape[0]
                                    
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




output_path = "../jsons/FL3D - Kaggle_1_euclidean_normalized.json"
with open(output_path, "w") as f:
    json.dump(output_data, f)
