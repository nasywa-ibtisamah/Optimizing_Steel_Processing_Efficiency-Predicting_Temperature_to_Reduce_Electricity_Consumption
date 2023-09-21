# Project Description

Hello there :wave:
I hope this project finds you well!

A plant aims to reduce its electricity consumption during the steel processing stage to lower production costs. This is achieved by determining the optimal temperature at the steel processing stage.

**Objective**

To determine the optimal temperature for metal during the steel processing stage, a regression model will be created to forecast the metal's temperature. The chosen model should exhibit a Mean Absolute Error (MAE) value below 6. The choice of the value 6 is based on the fact that an average error of 5-6 degrees in temperature prediction is considered acceptable in steel processing.

**Used Model**
- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- CatBoostRegressor
- XGBRegressor
- LGBMRegressor

# Methodology

Since the output of the model trained in this project is to predict metal temperature (continuous value), this falls into a regression problem, and an accurate and efficient **regression model** is needed.

Predicting a continuous value using a regression model involves several steps. Here's a general outline of the process:
1. Data Collection: As an assumption, I have all the datasets I need to start the project.
2. Data Preprocessing:
- Data Cleaning: standardize column names, drop unused columns, Handle missing values, fixing data types, outliers, and anomalies in your dataset.
- Feature Engineering: Select and transform relevant features that can help improve model accuracy. This involves creating new features.
- Train-Test Split: I will divide the final dataset into 2 datasets, train and test with a ratio of 75/25.
3. Selected Regression Models:
- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- CatBoostRegressor
- XGBRegressor
- LGBMRegressor
3. Model Training: Train the selected regression models on the training dataset. I will not utilise deep learning.
4. Model Evaluation: Evaluate the model's performance on the testing dataset using appropriate evaluation metrics. The used metric is Mean Absolute Error (MAE). The used scoring will be: 'neg_mean_absolute_error' (the scoring metric for cross-validation to negative mean absolute error (neg_MAE). In scikit-learn, it's common to use neg_MAE instead of MAE for scoring because scikit-learn optimizes for maximizing scores, but we typically want to minimize MAE. So, using the negative MAE allows the optimization process to work correctly. Additionally, I will utilise cross validation to assess the performance of a machine learning model and to reduce the risk of overfitting.The cross-validation score provides a more reliable estimate of the model's performance on unseen data compared to a single train-test split.
5. Hyperparameter Tuning: Fine-tune the model's hyperparameters to improve its performance. I will use grid search to find out the best params.
6. Prediction: Once my regression model is trained, I will use it to make predictions on new data points. Provide the model with the predictor variables, and it will output a continuous value as the prediction.

## 1. Data Overview

**Conclusion of Data Overview Stage**

There are several issues:

1. There are a lot of missing data in df_bulk, df_bulk_time df_wire df_wire_time, df_temp.
2. Missing values in df_bulk_time follow the same pattern as df_bulk.
3. Missing values in df_wire_time follow the same pattern as df_wire.
4. Incorrect data types > date and time values need to be converted to the datetime format and temperature colum should be converted to integer data type.
5. Inconsistent column names.

## 2. Data Engineering

   I will transform raw data into features that are more useful with adding new aggregate columns, as follows:

1. summarising sum_active_power and sum_reactive power, indexed by 'key'
2. exluding bad criteria of df_temp data
3. Finding start temperature and end temperature , based on 'min' and 'max' temperature time, indexed by 'key'


## 3. Data Preprocessing

Steps I did for processing the data:
1. Standardize Column Names
2. Drop Unused Columns
3. Prepare Final Dataset
4. Handling Missing Values
5. Fixing Data Types

## 4. Exploratory Data Analysis

**1. What is the correlation between features in the final dataset? What will be done with columns that have very low correlation values?**

**Findings**
1. The correlation coefficient between the features sum_active_power and sum_reactive_power is very high (0.96). I will remove sum_reactive_power from the model to avoid overfitting.
2. From the correlation coefficient values , out of 25 columns, it can be observed that the average correlation values are below 0.1 and also above -0.1. This indicates that the relationship between the target and other features is relatively weak.
3. Due to this condition, only features with correlation values greater than 0.1 and less than -0.1 will be used.
4. df_final is the final dataset that will be used during the training model and evaluation process.

**2. Are there any anomalous (outliers) data? What will be done to address the outliers?**

To detect the presence of anomalous or outlier data, the 'pyod' library will be used by importing the KNN algorithm. This algorithm can be used to identify data that is unusual or uncommon in a dataset.

**Findings and Action to take**
From the results, it can be seen that there are 196 anomalous/outlier data points. These data points will be dropped from the final dataset.

## 5. Model Training and Evaluation in Train Set

The neg_mean_absolute_error scoring is essentially the negative of the Mean Absolute Error (MAE). It's designed this way because scikit-learn's cross-validation and grid search functions aim to maximize scoring values, so they use the negation of error metrics to find the best model. A lower (more negative) neg_mean_absolute_error indicates better model performance.

 When I filtered the MAE value between -5.7 and -5.9, the top three models are as follows:

- CatBoostRegressor: 5.762954
- XGBRegressor: 5.777462
- LGBMRegressor: 5.894216

## 6. Applying the trained model to Test dataset

The Mean MAE value on the test set, obtained by applying the best model with the best parameters, is 5.39874692476624.

This model meets the minimum MAE value requirement (<= 8.7) and has an ideal MAE value < 6.

Furthermore, the close similarity in MAE values between the training set and the test set demonstrates that there is no overfitting in the model, indicating its accuracy.

# Discussion

## 1. Model Evaluation Report

Since the MAE values for all models meet the minimum requirement of <8.7, it can be said that the final dataset, which has undergone data preprocessing and exploratory data analysis (EDA), is already quite good, and the process is quite optimal. This can also be observed from the MAE values on the train set and test set, which are not significantly different, indicating the model is not overfitting.

## 2. Conclusions

This project has successfully trained a model capable of predicting the optimal end temperature of metal during steel processing to enhance energy efficiency and reduce production costs. The plant can utilize the CatBoostRegressor model with the known best parameters to predict the optimal end temperature. With an MAE value of 5.39, it indicates that the average error between the model's predictions and the actual values in the test data is approximately 5.39 degrees Celsius.

## 3. Recommendations

This project only uses one measurement metric, which is MAE. It would be better if other metrics, such as R2, MSE, RMSE, were also used for measurement.

  By using the RMSE metric, it is possible to measure the average error between the model's predictions and the actual values in the same units as the target variable. Additionally, R-squared does not provide information about how well the model fits the data.
