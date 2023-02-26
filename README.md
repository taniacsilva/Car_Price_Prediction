# Car_Price_Prediction

This project is inspired in a ML Zoomcamp.

In project I created a model to predict the price of a car. The dataset used was obtained from [kaggle](https://www.kaggle.com/CooperUnion/cardataset)

I have followed the following steps:

* Prepare data and Exploratory Data Analysis (EDA)

  **Main Conclusions** : The target variable distribution was transformed to a normal one, because before I had a long-tail distribution that usually can become an issue to ML models.


* Setting up the validation framework (split between train, validation and test)


* Use linear regression for predicting price


* Evaluating the model with RMSE


* Feature engineering  

**Main Conclusions** : Creation of features from existing ones, namely creation of the feature age of car. This feature improved significantly the performance of my model.


* Integration of categorical variables in the model

**Main Conclusions** : One-hot encoding was used


* Regularization

**Main Conclusions** : Adding a small number to the diagonal of the matrix (XTX) increased the performance of my model


* Using the model 

**Main Conclusions** : 


