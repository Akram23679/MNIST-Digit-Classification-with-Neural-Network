# MNIST-Digit-Classification-with-Neural-Network
Overview\
This notebook builds a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) from the MNIST dataset using TensorFlow/Keras. It covers the full ML pipeline: data loading, preprocessing, model building, training, evaluation, and deployment.

Libraries Used
1. numpy- Array operations and math
2. matplotlib/seaborn- Visualisation and plotting
3. tensorflow/keras- Model Buldling and training
4. sklearn- Evaluation metrics (confusion matrix, classification report)

Dataset
1. MNIST- 70,000 grayscale images of handwritten digits (0–9)
2. Training set- 60,000 images of shape (28,28)
3. Test set- 10,000 images of shape (28,28)
4. Classes- 10(digits 0 through 9)
5. pixel range (raw): [0,255]

Preprocessing
Steps
1. Normalise pixel values from [0, 255] → [0.0, 1.0] by dividing by 255
2. Add channel dimension: (28, 28) → (28, 28, 1) — required by Conv2D layers
3. One-hot encode labels: integer class labels → 10-element binary vectors using to_categorical

Model Architecture — MNIST_CNN
Input (28, 28, 1)
    ↓
Block 1: Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
    ↓
Block 2: Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
    ↓
Classifier: Flatten → Dense(256) → BN → ReLU → Dropout(0.5) → Dense(10, softmax)

Compilation
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

Data Augmentation
data_augmentation = keras.Sequential([
    layers.RandomRotation(factor=0.1),      # ±10% rotation
    layers.RandomZoom(height_factor=0.1),   # ±10% zoom
    layers.RandomTranslation(0.1, 0.1),     # ±10% pixel shift
])

tf.data Pipeline
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train_ohe))
    .shuffle(60_000)
    .batch(128)
    .map(lambda x, y: (data_augmentation(x, training=True), y),
         num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

<img width="1590" height="384" alt="1" src="https://github.com/user-attachments/assets/67958908-5ef2-401b-8f51-787f4c2a3353" />
<img width="1167" height="495" alt="2" src="https://github.com/user-attachments/assets/b85bf722-239a-4e50-b523-6cc80766e1c9" />
<img width="1388" height="495" alt="3" src="https://github.com/user-attachments/assets/a3317bce-7612-498b-a7ea-6a2afa3dd13c" />
<img width="931" height="790" alt="4" src="https://github.com/user-attachments/assets/cf227cfa-469c-46ba-a3c5-c20ac7034795" />
<img width="1389" height="1179" alt="5" src="https://github.com/user-attachments/assets/f5c96c2d-acdd-4abb-beb2-ce126e37bda0" />
<img width="1590" height="679" alt="6" src="https://github.com/user-attachments/assets/ab404089-cc2e-4ea9-98e8-a6d5a7874c63" />
<img width="1288" height="396" alt="10" src="https://github.com/user-attachments/assets/febcf2ac-d6f4-4321-8f0f-4e180524415d" />
<img width="1372" height="592" alt="9" src="https://github.com/user-attachments/assets/522a7d79-9ace-4419-9492-52185c0e775c" />
<img width="1285" height="396" alt="8" src="https://github.com/user-attachments/assets/aae2cb2a-5d70-4551-9215-f2bb8b47a080" />
<img width="910" height="390" alt="7" src="https://github.com/user-attachments/assets/3f101399-e6b9-437d-98fb-b799c39e1e4a" />

