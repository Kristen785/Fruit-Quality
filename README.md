# Fruit-Quality

📘 Project Overview

This project aims to develop a deep learning model that classifies the quality of fruits using image data from the Fruit Quality Datasets (FruQ-DB).
The project demonstrates the application of Convolutional Neural Networks (CNNs) and transfer learning (ResNet50) for image-based classification tasks.

👥 Group Members

#I have decided to be on my own as no one joined my group and therefore i decided to do he project on my own: 

Name	           Student Number	          Team Name
Kristen M. Hoff	 34292942	                 Arcane


🧩 Dataset Description


Dataset used: Fruit Quality Dataset (FruQ-DB)

Number of classes: Multiple fruit types and quality levels (e.g., Fresh, Rotten, Overripe).

Data type: RGB images.

Input shape: (224 × 224 × 3).

Target variable: Image label (fruit class/quality).


⚙️ Methodology

1️⃣ Data Preprocessing

First we need to load all images from class folders into a pandas DataFrame so that we can extract the information. After this we then need to split the dataset using stratified sampling. This can be in the form or ratio of 70% training, 15% validation and 15% testing. The most difficult but interesting part was to apply the ImageDataGenerator for real - time data augumentation. Despite this i still managed to make it work so we can get the data and see the data we need in order to successfully get all the values from our samples.


2️⃣ Model Architecture

This includes different training for the model. Firstly we need to train the header layers which is the freeze base model and then secondly we lower the learning rate by fine - tuning the last layers.


3️⃣ Evaluation Metrics:

The evaluation metrics we take into account in this project is accuracy, precision, recall, F1 - score, confusion matrix, ROC Curves and AUC.

📊 Results Summary:
The results can be summarised to show us the metric value and the score of each metric and are listed as follows:

Metric	                 Score
Training Accuracy     	~98%
Validation Accuracy   	~95%
Test Accuracy         	~94%
Macro AUC             	0.96
F1-Score              	0.93

🧾 Visualizations

Confusion Matrix: Displays correct and incorrect classifications.

ROC Curve: Evaluates multi-class discrimination power.

Training Curves: Loss and accuracy trends during epochs.

🧠 Discussion

The model achieved high accuracy and generalization, proving the effectiveness of transfer learning for visual quality classification tasks.
Minor misclassifications were mainly between similar fruit quality categories (e.g., ripe vs. overripe).
Further improvements could include fine-tuning deeper layers, using a larger dataset, or experimenting with models like EfficientNet or Vision Transformers (ViT).



🚀 How to Run
Requirements

Install dependencies:

pip install tensorflow scikit-learn pandas matplotlib seaborn pillow

Execution

Download and extract the FruQ-DB dataset from Zenodo

Set your dataset path in the notebook:

DATA_DIR = "/path/to/FruQ-DB"


Run the Jupyter notebook or Python script:

jupyter notebook FruQDB_CNN.ipynb


The model will train, evaluate, and save the best version as fruqd_resnet50_finetuned.h5.


🧩 Evaluation & Presentation

Evaluation based on:

Model accuracy and performance metrics

Code functionality and reproducibility

Oral presentation clarity and understanding


#📚 References

FruQ-DB Dataset: Zenodo Repository

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition (ResNet).

TensorFlow Documentation: https://www.tensorflow.org

