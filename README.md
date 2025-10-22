# Fruit-Quality

📘 Project Overview

This project aims to develop a deep learning model that classifies the quality of fruits using image data from the Fruit Quality Datasets (FruQ-DB).
The project demonstrates the application of Convolutional Neural Networks (CNNs) and transfer learning (ResNet50) for image-based classification tasks. <br>

👥 Group Members

I have decided to be on my own as no one joined my group and therefore i decided to do he project on my own. <br> 


Name: Kristen M. Hoff                            	                           
Student Number : 34292942	 <br>
Team Name : Arcane

                   	                                   
🧩 Dataset Description <br> <br>


**Dataset used:** Fruit Quality Dataset (FruQ-DB) <br>

**Number of classes:** Multiple fruit types and quality levels (e.g., Fresh, Rotten, Over ripe). <br>

**Data type:** RGB images. <br>

**Input shape:** (224 × 224 × 3 <br>

**Target variable:** Image label (fruit class/quality). <br>


⚙️ Methodology <br>


1️⃣ Data Preprocessing <br>


First we need to load all images from class folders into a pandas DataFrame so that we can extract the information. After this we then need to split the dataset using stratified sampling. This can be in the form or ratio of 70% training, 15% validation and 15% testing. The most difficult but interesting part was to apply the ImageDataGenerator for real - time data augumentation. Despite this i still managed to make it work so we can get the data and see the data we need in order to successfully get all the values from our samples.


2️⃣ Model Architecture <br>


This includes different training for the model. Firstly we need to train the header layers which is the freeze base model and then secondly we lower the learning rate by fine - tuning the last layers.


3️⃣ Evaluation Metrics <br>


The evaluation metrics we take into account in this project is accuracy, precision, recall, F1 - score, confusion matrix, ROC Curves and AUC.


📊 Results Summary <br>


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


🚀 How to Run The Project:


**Requirements**


**Install dependencies:**
1. pip install tensorflow scikit-learn pandas matplotlib seaborn pillow


**Execution**
2. Download and extract the FruQ-DB dataset from Zenodo

**Set your dataset path in the notebook:**
3. DATA_DIR = "/path/to/FruQ-DB"

**Run the Jupyter notebook or Python script:**
4. jupyter notebook FruQDB_CNN.ipynb


###The model is designed to train, evaluate, and save the best version of itself as fruqd_resnet50_finetuned.h5. This version will be the most accurate so far according to it's training and the end result.###


🧩 Evaluation


The evaluation is based on the model accuracy and performance metrics, code functionality and reproducibility.


#📚 References

FruQ-DB Dataset: Zenodo Repository

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition (ResNet).

TensorFlow Documentation: https://www.tensorflow.org

