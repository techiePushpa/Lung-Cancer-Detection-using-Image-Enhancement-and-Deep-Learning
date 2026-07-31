# 🫁 Lung Cancer Detection using Image Enhancement and Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Project Overview

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early diagnosis significantly improves treatment success and survival rates.

This project presents a **Deep Learning-based Lung Cancer Detection System** using CT scan images. Before classification, the CT images are enhanced using image preprocessing techniques to improve the visibility of important features.

The model classifies lung CT scan images into:

- Normal
- Benign
- Malignant

using a Convolutional Neural Network (CNN) integrated with an Attention Mechanism.

---

# 📂 Dataset

Dataset Used:

**IQ-OTHNCCD Lung Cancer Dataset**

Download from Kaggle:

https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset

After downloading, place the dataset inside:

```
dataset/
    normal/
    bengin/
    malignant/
```

---

# 🚀 Features

- Image Enhancement using CLAHE
- Brightness Adjustment
- CNN-based Deep Learning Model
- Attention Mechanism
- Three-Class Classification
- Accuracy & Loss Visualization
- Confusion Matrix
- Classification Report
- Sensitivity Calculation
- Specificity Calculation

---

# 🏗️ Project Structure

```
Lung-Cancer-Detection/
│
├── dataset/
│
├── notebooks/
│   └── Lung_Cancer_Detection.ipynb
│
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── metrics.py
│
├── models/
│
├── outputs/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Lung-Cancer-Detection.git
```

Go inside the project

```bash
cd Lung-Cancer-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

### Step 1

Download the dataset from Kaggle.

### Step 2

Extract it.

### Step 3

Place the folders inside

```
dataset/
```

The directory should look like

```
dataset/
    normal/
    bengin/
    malignant/
```

### Step 4

Run the training script

```bash
python src/train.py
```

or open

```
notebooks/Lung_Cancer_Detection.ipynb
```

and run all cells.

---

# 🧠 Model Architecture

Input CT Image

↓

Image Enhancement

- CLAHE
- Brightness Adjustment

↓

Convolutional Neural Network

↓

Attention Block

↓

Fully Connected Layer

↓

Softmax Classifier

↓

Prediction

---

# 📊 Experimental Results

## Test Accuracy

**97.73%**

---

## Training Performance

| Metric | Value |
|----------|--------|
| Training Accuracy | 93.96% |
| Validation Accuracy | 97.73% |
| Test Accuracy | 97.73% |
| Test Loss | 0.0965 |

---

# 📈 Training History

| Epoch | Train Accuracy | Validation Accuracy |
|--------|----------------|---------------------|
| 1 | 50.63% | 63.64% |
| 2 | 66.13% | 74.09% |
| 3 | 72.86% | 81.82% |
| 4 | 79.70% | 85.00% |
| 5 | 85.06% | 88.64% |
| 6 | 89.62% | 91.36% |
| 7 | 91.11% | 94.55% |
| 8 | 93.50% | 97.27% |
| 9 | 92.36% | 94.09% |
| 10 | 93.96% | 97.73% |

---

# 📉 Confusion Matrix

```
                 Predicted

               Normal Benign Malignant

Normal            88      0       1

Benign             4     15       0

Malignant          0      0      112
```

---

# 📋 Classification Summary

| Class | Precision | Recall | F1-Score |
|---------|-----------|----------|-----------|
| Normal | High | 98.88% | High |
| Benign | High | 78.95% | High |
| Malignant | High | 100% | High |

---

# 📌 Sensitivity & Specificity

| Class | Sensitivity | Specificity |
|----------|-------------|-------------|
| Normal | 98.88% | 96.95% |
| Benign | 78.95% | 100.00% |
| Malignant | 100.00% | 99.07% |

---

# 📷 Output Screenshots

Save the following images inside

```
outputs/
```

- accuracy.png
- confusion_matrix.png
- prediction.png

Then display them:

```markdown
## Accuracy Graph

![Accuracy](outputs/accuracy.png)

## Confusion Matrix

![Confusion Matrix](outputs/confusion_matrix.png)
```

---

# 💡 Future Improvements

- Transfer Learning (EfficientNet, ResNet50)
- Vision Transformer (ViT)
- Explainable AI using Grad-CAM
- Flask Web Application
- Streamlit Deployment
- Mobile Application
- Multi-class Disease Detection

---

# 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab

---

# 👩‍💻 Author

**Your Name**

B.Tech Computer Science Engineering

GitHub: https://github.com/yourusername

LinkedIn: https://linkedin.com/in/yourprofile

---

# ⭐ If you found this project useful, please consider giving it a star!
