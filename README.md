# 🌿 Disease Identification in Maize Leaf Using CNN

This project aims to identify and classify diseases in maize leaves using **Convolutional Neural Networks (CNN)**. By analyzing leaf images, the model predicts the type of disease affecting the maize crop, enabling timely intervention for farmers.

## 🚀 Demo
![App Screenshot](Streamlit%20Web%20Interface.png)

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

   git clone https://github.com/yourusername/Disease_Identification_In_Maize_Leaf_Using_CNN.git
   cd Disease_Identification_In_Maize_Leaf_Using_CNN
3. **Install required libraries:**
   pip install -r requirements.txt
4. **Run the Streamlit app:**
   streamlit run app.py

## 🧠 Model Architecture
1. **Base Model:** Custom CNN.
2. **Layers:**
    - Convolutional Layers
    - Batch Normalization
    - Max Pooling
    - Fully Connected Layers
3. **Activation Function:** ReLU for hidden layers, Softmax for output.
4. **Optimizer:** Adam
5. **Loss Function:** Categorical Cross-Entropy

## 📊 Results
| Metric     | Value  | 
|------------|--------|
| Accuracy   | 98.00%  | 
| Precision  | 97.60%  | 
| Recall     | 97.80%  | 
| F1-Score   | 98.00%  | 


## 📌 Key Code Snippet
```
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))    
   
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))    
   
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))    
    
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))    

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
          
model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=["accuracy"])
```

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
