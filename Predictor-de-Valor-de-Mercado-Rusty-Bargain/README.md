# Predicting Market Value for Rusty Bargain Used Cars

## Project Overview

This project focuses on developing a machine learning model to predict the market value of used cars for the Rusty Bargain platform. The application aims to help users quickly determine the market value of their vehicles by leveraging historical data on car specifications, equipment versions, and pricing.

### Key Objectives:
- **Prediction Accuracy:** Ensure high-quality predictions that align with market values.
- **Prediction Speed:** Optimize the model for fast and efficient predictions.
- **Training Time:** Minimize the time required to train the model without sacrificing accuracy.

## Project Structure

1. **Data Preparation:**
   - Data cleaning and feature engineering to handle missing values, remove duplicates, and encode categorical variables.
   - Scaling numerical features to improve model performance.

2. **Model Training:**
   - Implementation of various regression models, including `RandomForestRegressor`, `CatBoostRegressor`, `LightGBMRegressor`, and `XGBoostRegressor`.
   - Hyperparameter tuning to optimize model performance.

3. **Model Evaluation:**
   - Comparison of models based on RMSE (Root Mean Square Error) and computational efficiency.
   - Analysis of model predictions to ensure they align with the real-world market values.

## Conclusion

The project successfully developed a predictive model for estimating the market value of used cars. The `LightGBMRegressor` provided the best balance between speed and accuracy, making it the optimal choice for deployment in the Rusty Bargain application.

## Installation and Usage

1. Clone the repository to your local machine.
2. Install the required Python libraries using `pip install -r requirements.txt`.
3. Run the Jupyter notebook to train the model and evaluate the results.

## Future Work

Future improvements could include experimenting with additional features, such as regional price variations and seasonal trends, to further enhance model accuracy.

## Dataset link
https://practicum-content.s3.us-west-1.amazonaws.com/datasets/car_data.csv