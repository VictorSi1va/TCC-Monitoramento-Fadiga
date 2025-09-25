import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import onnxruntime as ort

# ==== Parâmetros ====
WINDOW_SIZE = 45
MODEL_PATH = "./create_labels/code/documentos/model_lstm_victorv5.onnx"

# ==== Inicializar MediaPipe ====
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ==== Índices dos keypoints usados ====
FACE_IDX = {
    "mouth_outer":   [61, 40, 37, 0, 267, 270, 291],         # boca externa (pontos principais)
    "mouth_inner":   [78, 95, 14, 317, 308],                 # boca interna (centro e extremos)
    "left_eye":      [33, 160, 158, 133, 153, 144, 145],     # contorno da pálpebra esquerda
    "right_eye":     [362, 385, 387, 263, 373, 380, 381],    # contorno da pálpebra direita
    "left_brow":     [70, 105, 107],                         # 3 pontos-chave
    "right_brow":    [336, 334, 300],
    "nose":          [1, 2, 168],                            # ponta + centro
    "chin":          [152]                                   # queixo central
}

all_landmark_indices = sorted(set(idx for group in FACE_IDX.values() for idx in group))

# ==== ONNX Runtime ====
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# ==== Janela de features ====
window = deque(maxlen=WINDOW_SIZE)





def euclidean(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

left_eye_idx = 33
right_eye_idx = 263




# ==== Webcam ====
cap = cv2.VideoCapture(0)


# ==== Vídeo Input ====
# input_video_path = "./create_labels/datasets/Videos Fadiga/gB_10_s5_2019-03-12T10_35_20+01_00_rgb_face.mp4"
# output_video_path = "./create_labels/datasets/Videos Fadiga/gB_10_s5_50sec.mp4"
# cap = cv2.VideoCapture(input_video_path)

# fps = cap.get(cv2.CAP_PROP_FPS)
# width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# num_frames_to_save = int(fps * 50)

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Ou 'XVID'
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# ------------------------------------------



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    keypoints = []
    
    
    
    #EUCLIDEAN DISTANCE
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        cx, cy = landmarks[1].x, landmarks[1].y  # Nariz
    
        # Distância entre os cantos dos olhos
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
                keypoints.extend([x, y])
    
                # Apenas para visualização (se quiser)
                px = int(lm.x * image.shape[1])
                py = int(lm.y * image.shape[0])
                cv2.circle(image, (px, py), 2, (0, 255, 0), -1)
    
            else:
                keypoints.extend([0.0, 0.0])
                
                
    # if results.multi_face_landmarks:
    #     landmarks = results.multi_face_landmarks[0].landmark



    #     #NARIZ
    #     h, w = image.shape[:2]

    #     nose = landmarks[1]
    #     cx, cy = nose.x, nose.y
        
    #     for idx in all_landmark_indices:
    #         if idx < len(landmarks):
    #             lm = landmarks[idx]
    #             x = lm.x - cx
    #             y = lm.y - cy
        
    #             # Convertendo para coordenadas de pixel APENAS para visualização
    #             px = int((lm.x) * w)
    #             py = int((lm.y) * h)
    #             cv2.circle(image, (px, py), 2, (0, 255, 0), -1)
        
    #             keypoints.extend([x, y])
                
        
        
    #     #NORMAL
    #     for idx in all_landmark_indices:
    #         if idx < len(landmarks):
    #             x = int(landmarks[idx].x * image.shape[1])
    #             y = int(landmarks[idx].y * image.shape[0])
    #             cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    #             keypoints.extend([landmarks[idx].x,
    #                               landmarks[idx].y])
    #         else:
    #             keypoints.extend([0.0, 0.0])
    else:
        keypoints = [0.0, 0.0] * len(all_landmark_indices)

    window.append(keypoints)
    
    if len(window) == WINDOW_SIZE:
        x_input = np.array([window], dtype=np.float32)
        prediction = session.run(None, {input_name: x_input})[0][0]
        predicted_class = int(np.argmax(prediction)) if prediction.shape[-1] > 1 else int(prediction > 0.5)
        
        if predicted_class != 0:
            print(predicted_class)
        label_str = ["Alerta", "Bocejando", "Microsleep"][predicted_class]
    else:
        label_str = "Carregando..."

    # Mostrar o estado previsto
    cv2.putText(image, f"Estado: {label_str}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)

    cv2.imshow("Webcam - Detecção de Fadiga (ONNX)", image)
    
    # out.write(image)

    if cv2.waitKey(1) & 0xFF == 27:  # Tecla ESC
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
