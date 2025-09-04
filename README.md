# 💤 Bocejos LSTM – Detecção de Sonolência com LSTM

![GIF](./documentos/GIF_GITHUB.gif)

## 🛠️ Tecnologias Utilizadas

- Python 3.9  
- TensorFlow  
- ONNX  
- MediaPipe  
- OpenCV  
- NumPy  
- Conda (ambientes virtuais)

---

Este projeto utiliza modelos LSTM para classificar estados de sonolência com base em keypoints extraídos de vídeos. As classes detectadas são:

- **0 - Alert**: Olhos abertos e postura estável  
- **1 - Yawning**: Boca se abrindo com expressão clara  
- **2 - Microsleep**: Olhos fechando por tempo prolongado  

## 📁 Estrutura do Repositório

```bash
Bocejos LSTM/
│
├── create_labels/                        # Scripts e dados para criação de labels e treinamento do modelo
│   ├── code/                             # Código-fonte principal
│   │   ├── documentos/                   # Modelo LSTM salvo em formato ONNX
│   │   │   └── model_lstm.onnx
│   │   ├── model_tensorflow/            # Modelos e scripts em TensorFlow (base para exportação ONNX)
│   │   │   ├── extract_keypoints-kaggle.py              # Extrai keypoints do dataset Kaggle (sem mãos)
│   │   │   ├── extract_keypoints-kaggle-hands.py        # Extrai keypoints do dataset Kaggle (com mãos)
│   │   │   ├── extract_keypoints-videos.py              # Extrai keypoints de vídeos locais (sem mãos)
│   │   │   ├── extract_keypoints-videos-hands.py        # Extrai keypoints de vídeos locais (com mãos)
│   │   │   ├── train_lstm.py                            # Treinamento do modelo LSTM (sem mãos)
│   │   │   └── train_lstm-hands.py                      # Treinamento do modelo LSTM (com mãos)
│   ├── datasets/                        # Datasets utilizados no projeto (não incluídos por tamanho)
│   │   ├── FL3D - Kaggle
│   │   ├── DMD – Driving Monitoring Dataset
│   │   └── Videos Fadiga - Pessoal
│   ├── jsons/                           # JSONs com keypoints extraídos
│   │       ├── FL3D - Kaggle3.json
│   │       ├── DMD – Driving Monitoring Dataset.json
│   │       ├── Videos pessoais para fine-tuning do modelo.json
│   │       └── Other/
│   ├── labels_explanation.txt          # Explicação das labels utilizadas para classificação
│   ├── save_to_onnx.txt                # Comando de exemplo para exportar o modelo para formato ONNX
│
├── documentos/                         # Pasta genérica para arquivos auxiliares
├── run.py                              # Executa o modelo em tempo real via webcam (sem mãos)
├── run-hands.py                        # Executa o modelo em tempo real via webcam (com mãos)
├── requirements.txt                    # Dependências para treinamento e execução do modelo padrão
├── requirements_onnx.txt               # Dependências adicionais para exportação do modelo para ONNX
```

---

## ℹ️ Observação sobre os arquivos com -hands no nome:
Os arquivos que possuem -hands no nome são versões alternativas dos scripts que incluem keypoints das mãos durante a extração dos dados e no treinamento do modelo. Essa variação pode ser útil para cenários onde o movimento das mãos é relevante para a detecção de bocejos ou fadiga.

Por exemplo:

`train_lstm.py`: treinar modelo sem considerar mãos
`train_lstm-hands.py`: treinar modelo considerando keypoints das mãos

`run.py`: inferência em tempo real sem mãos
`run-hands.py`: inferência em tempo real com mãos

Escolha os scripts com ou sem hands dependendo da abordagem desejada para o seu caso de uso.

---

## 📦 Como Instalar e Executar

### 1️⃣ Ambiente principal (treinar modelo e executar via webcam)

```bash
conda create -n lstm_env python=3.9 -y
conda activate lstm_env

pip install -r requirements.txt
```

### 2️⃣ Ambiente ONNX (converter modelo para ONNX)

```bash
conda create -n lstm_env_onnx python=3.9 -y
conda activate lstm_env_onnx

pip install -r requirements_onnx.txt
```

---

## 🚀 Como Usar

### 1. Extrair keypoints:
- Para o dataset do **Kaggle**:
  ```bash
  python extract_keypoints-kaggle.py
  ```
- Para vídeos locais:
  ```bash
  python extract_keypoints-videos.py
  ```

### 2. Treinar o modelo:
```bash
python train_lstm.py
```
O modelo será salvo nas pastas `model_tensorflow/` e `code/documentos/`.

### 3. Converter para ONNX:
Execute o comando dentro de `save_to_onnx.txt` no terminal, dentro do ambiente `lstm_env_onnx`.

### 4. Rodar o modelo em tempo real (via webcam):
```bash
python run.py
```

---

## 🧠 Labels Classificadas

| Classe      | Descrição                                 | Valor |
|-------------|--------------------------------------------|--------|
| `alert`     | Olhos abertos e postura estável           | `0`    |
| `yawning`   | Boca se abrindo com expressão clara       | `1`    |
| `microsleep`| Olhos fechando por tempo prolongado       | `2`    |

---

## 📌 Observações

- Os datasets utilizados estão na pasta `datasets/`, mas **não foram incluídos no repositório** devido ao tamanho.  
- Os arquivos JSON com os keypoints extraídos (que podem ser usados para treinamento na mesma pasta) estão em `jsons/`.

## Resultados e Decisões

Testamos dois modelos: um usando apenas pontos faciais e outro incluindo também os keypoints das mãos. O modelo **sem mãos** teve desempenho melhor. A inclusão das mãos piorou os resultados, principalmente porque as mãos muitas vezes tampam a boca, e como temos poucos dados, o modelo acaba interpretando qualquer mão na frente como bocejo.

### Comparativo:

- **Modelo sem mãos:**

  ![Sem mãos](./documentos/lstm_euclidean_normalized.png)

- **Modelo com mãos:**

  ![Com mãos](./documentos/lstm_hands_euclidean_normalized.png)

Ambos foram treinados com ~86.000 frames (20% para teste). O modelo sem mãos teve um equilíbrio melhor entre precisão e recall, especialmente para a classe "Bocejando", que era a mais sensível a erros.

---

## Normalização Euclidiana

A melhor forma de pré-processamento que encontramos foi a **normalização euclidiana** dos keypoints faciais. Basicamente, pegamos a distância entre os olhos e usamos ela para normalizar a posição dos pontos, assim o modelo não é afetado por variações de distância da pessoa até a câmera.

Fórmula usada:

```python
def euclidean(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
```

Keypoints de referência:
```python
left_eye_idx = 33
right_eye_idx = 263
```

---

## Arquitetura do Modelo

```python
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))
```

Foi usado `Masking` para ignorar padding com zeros, uma camada LSTM para processar a sequência dos frames e duas densas para a classificação.

---

## Treinamento

O modelo foi treinado com early stopping ativado, para evitar overfitting:

```python
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```

```python
model.fit(X_train, y_train,
          epochs=50,
          batch_size=32,
          validation_data=(X_test, y_test),
          # class_weight=class_weights,
          callbacks=[early_stop])
```

---

## Dados Utilizados

- **FL3D (Kaggle)**: 49 GIFs de bocejos
- **6 vídeos adicionais** (Google Drive): gravações com eventos de fadiga

Os vídeos foram convertidos em frames e os keypoints foram extraídos com MediaPipe.

---
