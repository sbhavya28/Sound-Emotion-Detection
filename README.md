# 🎙️ Sound Emotion Detection

This project focuses on **speech emotion recognition** using deep learning. It is trained on the **TESS (Toronto Emotional Speech Set)** dataset and achieves an impressive **accuracy of 96.85%** on test data.

> 🚧 **Note:** Real-time emotion detection is **not currently supported** in this version. It will be implemented soon in upcoming updates.

---

## 📌 Dataset

- **Name:** [Toronto Emotional Speech Set (TESS)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
- **Classes:** 7 emotions — `angry`, `disgust`, `fear`, `happy`, `neutral`, `ps`, `sad`

---

## 🧠 Model Architecture

The model is a **deep fully connected neural network** built using TensorFlow Keras:

```python
model = Sequential([
    Dense(256, activation='relu', input_shape=(51,)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(7, activation='softmax')  # 7 emotion classes
])
```
Loss Function: Sparse Categorical Crossentropy

Optimizer: Adam (Learning rate = 0.001)

Final Accuracy: 0.9685714244842529 (~96.86%)

## 🎧 Feature Extraction
Features are extracted from .wav files using librosa and soundfile:

MFCCs (Mel Frequency Cepstral Coefficients)

Delta & Delta-Delta

Chroma Frequencies

```python
def extract_features(filename):
    ...
    mfccs = librosa.feature.mfcc(...)
    delta_mfcc = librosa.feature.delta(mfccs)
    delta2_mfcc = librosa.feature.delta(mfccs, order=2)
    chroma = librosa.feature.chroma_stft(...)
    ...
    return np.hstack([...])
```
🎨 Note: Spectrograms were not used in this version — features are purely numerical for MLP compatibility.

## 🛠️ Requirements
TensorFlow

Librosa

Soundfile

NumPy

scikit-learn

Install using:

```bash
pip install tensorflow librosa soundfile numpy scikit-learn
```

## 🚀 Future Improvements
✅ Add support for real-time audio input and prediction

🧠 Build a CNN version using spectrogram images

🌐 Deploy as a web app using Streamlit or Gradio

🌍 Extend to multilingual and larger datasets

## 📂 Folder Structure
```bash
Sound-Emotion-Detection/
│
├── main.ipynb             # Model training and evaluation
├── emotion_model.h5       # Trained model
├── README.md              # Project overview
└── audio_samples/         # Recorded or test audio samples
```
## 🧑‍💻 Author
Bhavya Shukla
For contributions, bug reports, or feedback, feel free to connect via GitHub or [LinkedIn](https://www.linkedin.com/in/bhavya-shukla-6782a8268/)

