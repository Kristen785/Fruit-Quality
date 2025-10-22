# Fruit-Quality

📘 Project Overview

This project aims to develop a deep learning model that classifies the quality of fruits using image data from the Fruit Quality Datasets (FruQ-DB).
The project demonstrates the application of Convolutional Neural Networks (CNNs) and transfer learning (ResNet50) for image-based classification tasks. <br>

👥 Group Members

I have decided to be on my own as no one joined my group and therefore i decided to do he project on my own. <br> 


Name: Kristen M. Hoff                            	                           
Student Number : 34292942	 <br>
Team Name : Arcane <br>

                   	                                   
🧩 Dataset Description <br> <br>


**Dataset used:** Fruit Quality Dataset (FruQ-DB) <br> <br>

**Number of classes:** Multiple fruit types and quality levels (e.g., Fresh, Rotten, Over ripe). <br> <br>

**Data type:** RGB images. <br> <br>

**Input shape:** (224 × 224 × 3 <br> <br>

**Target variable:** Image label (fruit class/quality). <br> <br>


⚙️ Methodology <br> <br> <br>


1️⃣ Data Preprocessing <br><br>


First we need to load all images from class folders into a pandas DataFrame so that we can extract the information. After this we then need to split the dataset using stratified sampling. This can be in the form or ratio of 70% training, 15% validation and 15% testing. The most difficult but interesting part was to apply the ImageDataGenerator for real - time data augumentation. Despite this i still managed to make it work so we can get the data and see the data we need in order to successfully get all the values from our samples. <br> <br>


2️⃣ Model Architecture <br><br>


This includes different training for the model. Firstly we need to train the header layers which is the freeze base model and then secondly we lower the learning rate by fine - tuning the last layers.<br> <br>


3️⃣ Evaluation Metrics <br> <br>


The evaluation metrics we take into account in this project is accuracy, precision, recall, F1 - score, confusion matrix, ROC Curves and AUC. <br> <br>


📊 Results Summary <br> <br> 

The results can be summarised to show us the metric value and the score of each metric and are listed as follows: <br> <br> 

Metric                       Score <br> 
Training Accuracy          	~98% <br> 
Validation Accuracy       	~95% <br> 
Test Accuracy             	~94% <br> 
Macro AUC                 	0.96 <br> 
F1-Score                  	0.93 <br> 


🧾 Visualizations <br>  <br> 


Confusion Matrix: Displays correct and incorrect classifications. <br> <br> 

ROC Curve: Evaluates multi-class discrimination power. <br> <br> 

Training Curves: Loss and accuracy trends during epochs.<br> <br> 


🧠 Discussion <br> <br> 

The model achieved high accuracy and generalization, proving the effectiveness of transfer learning for visual quality classification tasks. <br>  <br> 
Minor misclassifications were mainly between similar fruit quality categories (e.g., ripe vs. overripe).<br>  <br> 
Further improvements could include fine-tuning deeper layers, using a larger dataset, or experimenting with models like EfficientNet or Vision Transformers (ViT). <br>  <br> 


🚀 How to Run The Project: <br>  <br> 


**Requirements** <br> <br>  

1. pip install tensorflow scikit-learn pandas matplotlib seaborn pillow <br>


**Execution** <br> <br>

2. Download and extract the FruQ-DB dataset from Zenodo <br>


**Set your dataset path in the notebook:** <br> <br>

3. DATA_DIR = "/path/to/FruQ-DB" <br>


**Run the Jupyter notebook or Python script:** <br> <br>

4. jupyter notebook FruQDB_CNN.ipynb <br>


**The model is designed to train, evaluate, and save the best version of itself as fruqd_resnet50_finetuned.h5. This version will be the most accurate so far according to it's training and the end result.** <br>


🧩 **Evaluation** <br> <br>


The evaluation is based on the model accuracy and performance metrics, code functionality and reproducibility. <br> <br> <br>


📚 References <br> <br>


FruQ-DB Dataset: Zenodo Repository <br>

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition (ResNet). <br>

TensorFlow Documentation: https://www.tensorflow.org <br>

