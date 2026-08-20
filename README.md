# 🔢 MNIST Digit Classification with Neural Network

A Convolutional Neural Network (CNN) built with **TensorFlow/Keras** that classifies handwritten digits (0–9) from the MNIST dataset — achieving **99.46% test accuracy**. The notebook covers the full deep learning pipeline: data loading, preprocessing, augmentation, model building, training with callbacks, evaluation, and inference on custom uploaded images.

## 📌 Overview

MNIST is the classic "hello world" of computer vision. This project goes beyond a bare-bones classifier by adding batch normalization, dropout regularization, data augmentation, an efficient `tf.data` input pipeline, and training callbacks — resulting in a model that misclassifies only **54 out of 10,000** test images.

## 📂 Dataset

**MNIST** — 70,000 grayscale images of handwritten digits, loaded via `keras.datasets.mnist`:

| Split | Images | Shape |
|---|---|---|
| Training | 60,000 | (28, 28) |
| Test | 10,000 | (28, 28) |

- **Classes:** 10 (digits 0–9)
- **Raw pixel range:** [0, 255]

## ⚙️ Preprocessing

1. **Normalize** pixel values from `[0, 255]` → `[0.0, 1.0]`
2. **Add channel dimension** — `(28, 28)` → `(28, 28, 1)`, required by `Conv2D` layers
3. **One-hot encode** integer labels into 10-element binary vectors via `to_categorical`

## 🏗️ Model Architecture — `MNIST_CNN`

```
Input (28, 28, 1)
    ↓
Block 1: Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
    ↓
Block 2: Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
    ↓
Classifier: Flatten → Dense(256) → BN → ReLU → Dropout(0.5) → Dense(10, softmax)
```

- **Optimizer:** Adam (learning rate `1e-3`)
- **Loss:** Categorical cross-entropy
- **Total parameters:** 872,426 (871,530 trainable)

## 🔄 Data Augmentation

Applied on-the-fly during training to improve generalization:

```python
data_augmentation = keras.Sequential([
    layers.RandomRotation(factor=0.1),      # ±10% rotation
    layers.RandomZoom(height_factor=0.1),   # ±10% zoom
    layers.RandomTranslation(0.1, 0.1),     # ±10% pixel shift
])
```

## ⚡ Efficient Input Pipeline

Training data flows through a `tf.data.Dataset` pipeline (shuffle → batch → augment → prefetch) for fast, non-blocking training:

```python
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train_ohe))
    .shuffle(60_000)
    .batch(128)
    .map(lambda x, y: (data_augmentation(x, training=True), y),
         num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)
```

## 🎯 Training

Trained for up to 30 epochs with three callbacks to keep training efficient and robust:

- **EarlyStopping** — stops when `val_loss` stalls (patience 8), restores best weights
- **ReduceLROnPlateau** — halves the learning rate when `val_loss` plateaus (patience 4)
- **ModelCheckpoint** — saves the best model by `val_accuracy`

## 📊 Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **99.46%** |
| Test Loss | 0.0172 |
| Misclassified | 54 / 10,000 (0.54% error rate) |
| Total parameters | 872,426 |

Evaluation includes a full confusion matrix, per-class classification report, and visualizations of both correct and misclassified predictions.

## 🖼️ Inference on Your Own Images

The notebook includes a full pipeline for testing the model on custom, hand-drawn digit images:

1. Upload any image (via Colab's file picker)
2. Convert to grayscale, resize to 28×28, normalize, and invert if needed (MNIST digits are white-on-black)
3. Get a prediction with confidence score, shown alongside the processed image

## 🛠️ Tech Stack

- **Python 3**
- **TensorFlow / Keras** — model building, training, and inference
- **NumPy** — array operations
- **Matplotlib** & **Seaborn** — visualization (training curves, confusion matrix, sample predictions)
- **scikit-learn** — evaluation metrics (`classification_report`, `confusion_matrix`)
- **Pillow (PIL)** — custom image preprocessing
- **Google Colab** — development environment (file upload widget)

## 📁 Repository Structure

```
.
├── MNIST_Digit_Classification_with_Neural_Network.ipynb   # Full pipeline: data, model, training, evaluation, inference
└── README.md                                               # Project documentation
```

## 🚀 Getting Started

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/MNIST-Digit-Classification-with-Neural-Network.git
   cd MNIST-Digit-Classification-with-Neural-Network
   ```
2. Install dependencies
   ```bash
   pip install tensorflow numpy matplotlib seaborn scikit-learn pillow
   ```
3. Open and run the notebook
   ```bash
   jupyter notebook MNIST_Digit_Classification_with_Neural_Network.ipynb
   ```
   *(The custom-image upload cells use `google.colab.files` and are best run in Google Colab; the rest of the notebook runs anywhere.)*

## 📈 Key Takeaways

- Batch normalization + dropout gave a strong regularization boost, helping the model generalize well beyond 99% test accuracy.
- Data augmentation (rotation, zoom, translation) improved robustness to natural handwriting variation.
- `EarlyStopping` + `ReduceLROnPlateau` avoided overfitting while keeping training efficient.
- The model performs well even on out-of-distribution, user-drawn digit images once properly preprocessed (grayscale, resized, inverted).

## 📄 License

This project is open source and available for educational and research purposes.
