# Car Price Prediction

This project is inspired in a ML Zoomcamp.

In this project I created a model to predict the price of a car. The dataset used was obtained from [kaggle](https://www.kaggle.com/CooperUnion/cardataset).

I have followed the following steps:

* 👀 Prepare data and Exploratory Data Analysis (EDA)

  **Main Conclusions** : The target variable distribution was transformed to a normal one, because before I had a long-tail distribution that usually can become an issue to ML models.


* 🐱‍👤 Setting up the validation framework (split between train, validation and test)

   **Main Conclusions** : For each partition, feature matrices (X) and y vectors of targets were obtained. I have calculated the size of partitions and records are shuffled to guarantee that values of the three partitions contain non-sequential records of the dataset, and the partitions are created with the shuffled indices.


* 👩‍💻 Use linear regression for predicting price

   **Main Conclusions** : Obtaining predictions as close as possible to target values requires the calculation of weights from the general LR equation

   Normal Equation : w = $(X^T*X)^{-1}*X^T*y$
   


* ✔ Evaluating the model with RMSE


* 🏋️‍♀️ Feature engineering  

   **Main Conclusions** : Creation of features from existing ones, namely creation of the feature age of car. This feature improved significantly the performance of my model.


* 👨‍🚀 Integration of categorical variables in the model

   **Main Conclusions** : One-hot encoding was used


* 📏 Regularization

   **Main Conclusions** : Adding a small number to the diagonal of the matrix (XTX) increased the performance of my model


* 🎆 Using the model 

   **Main Conclusions** : 


