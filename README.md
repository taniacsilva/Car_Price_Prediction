# Car Price Prediction

This project is inspired in a ML Zoomcamp.

In this project I created a model to predict the price of a car. The dataset used was obtained from [kaggle](https://www.kaggle.com/CooperUnion/cardataset).

I have followed the following steps:

* 👀 Prepare data and Exploratory Data Analysis (EDA)

  **Main Conclusions** : The target variable distribution was transformed to a normal one, because before I had a long-tail distribution that usually can become an issue to ML models.


* 🐱‍👤 Setting up the validation framework (split between train, validation and test)

   **Main Conclusions** : For each partition, feature matrices (X) and y vectors of targets were obtained. I have calculated the size of partitions and records are shuffled to guarantee that values of the three partitions contain non-sequential records of the dataset, and the partitions are created with the shuffled indices.


* 👩‍💻 Use linear regression for predicting price

   **Main Conclusions** : Obtaining predictions as close as possible to target values requires the calculation of weights from the general LR equation. The feature matrix does not have an inverse because it is not square, so it is required to obtain an approximate solution, which can be obtained using the Gram matrix. The vector of weights or coefficients obtained with this formula is the closest possible solution to the LR system.
   Some of the features in x_train are NaN. So, I set them to 0, to the model be solvable. Please notice that other non-zero values can be used as filler (e.g. mean or I could use KNN algorithm).

   Normal Equation :  $w=(X^TX)^{-1}X^Ty$

    Where, $X^TX$ is the Gram Matrix

* ✔ Evaluating the model with RMSE
    **Main Conclusions** : I have taken some visual comparisons by plotting predicted y and the actual y using an histogram. I have also used RMSE in order to quantify how good or bad the model is. RMSE measures the error associated with the model being evaluated and enables to compare models.

    $RMSE=\sqrt{frac{1}/{m}}$

* 🏋️‍♀️ Feature engineering  

   **Main Conclusions** : Creation of features from existing ones, namely creation of the feature age of car. This feature improved significantly the performance of my model.


* 👨‍🚀 Integration of categorical variables in the model

   **Main Conclusions** : One-hot encoding was used


* 📏 Regularization

   **Main Conclusions** : Adding a small number to the diagonal of the matrix (XTX) increased the performance of my model


* 🎆 Using the model 

   **Main Conclusions** : 


