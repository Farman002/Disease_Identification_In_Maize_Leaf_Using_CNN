# 🌿 Disease Identification in Maize Leaf Using CNN

This project aims to identify and classify diseases in maize leaves using **Convolutional Neural Networks (CNN)**. By analyzing leaf images, the model predicts the type of disease affecting the maize crop, enabling timely intervention for farmers.

## 🚀 Demo
![App Screenshot](Streamlit Web Interface.png)

## 📋 Features
- **Image Classification:** Detects and classifies multiple types of diseases in maize leaves.
- **Deep Learning:** Uses **CNN (Convolutional Neural Network)** architecture for high accuracy.
- **Data Augmentation:** Enhances model performance by creating variations of existing images.
- **User Interface:** Streamlit app for easy image upload and disease prediction.
- **Accuracy:** Achieves high precision in identifying diseases.

## 🗂 Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/farmannaim/maizeleaf)  
- **Classes:** Healthy, Common Rust, Gray Leaf Spot, Northern Leaf Blight, etc.
- **Size:** Contains thousands of labeled images.
- **Preprocessing:** Resized images, normalization, and data augmentation applied.

## 🛠️ Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Disease_Identification_In_Maize_Leaf_Using_CNN.git
   cd Disease_Identification_In_Maize_Leaf_Using_CNN
2. **Install required libraries:**
   '''bash
   pip install -r requirements.txt
3. **Run the Streamlit app:**
   '''bash
   streamlit run app.py

## 🧠 Model Architecture
1. **Base Model:** MobileNetV2 / ResNet (replace based on your choice).
2. **Layers:**
    - Convolutional Layers
    - Batch Normalization
    - Max Pooling
    - Fully Connected Layers
3. **Activation Function:** ReLU for hidden layers, Softmax for output.
4. **Optimizer:** Adam
5. **Loss Function:** Categorical Cross-Entropy

## 📊 Results
- **Metric**	**Value**
- **Accuracy**	**95.0%**
- **Precision**	**94.5%**
- **Recall**	**93.8%**
- **F1-Score**	**94.1%**

## 📌 Key Code Snippet
- model = Sequential([
-     Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
-    MaxPooling2D((2, 2)),
-    Flatten(),
-    Dense(128, activation='relu'),
-    Dense(5, activation='softmax')  
- ])
- model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

## ⚙️ Requirements
1. **Python 3.8+**
2. **TensorFlow**
3. **Keras**
4. **OpenCV**
5. **Streamlit**
6. **NumPy**
7. **Pandas**

## 📞 Contact
- **Author:** Farman Naim
- **E-Mail:** farmannaim@gmail.com
- **LinkedIn:** https://www.linkedin.com/in/farman-naim/

## ⭐ Acknowledgments
- **Kaggle:** For the dataset.
- **TensorFlow and Keras:** For deep learning support.
- **Streamlit:** For the user interface.
