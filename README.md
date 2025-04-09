# Econometrics-Price-Prediction

## Abstract
This study focuses on predicting car selling prices using regression models, including Linear Regression, Polynomial Regression, and regularized models such as Ridge and Lasso. The goal was to select the most effective model based on performance metrics like MAE, MSE, RMSE, and R2 score. The dataset was cleaned and preprocessed, and various assumptions related to linearity, homoscedasticity, and normality were tested. The results revealed that Polynomial Regression (Degree 2) provided the best balance between model complexity and predictive accuracy. This report details the methodology, analysis, and key findings from the study, offering actionable insights for future improvements.

## 2. Methodology

### 2.1. Dataset
The dataset consists of several attributes related to used cars, including features such as brand, model, year, mileage, engine size, and more. The target variable for this regression task was the Selling Price.

### 2.2. Analytical Methods
The following steps were taken to ensure accurate model development:
- **Data Cleaning:** Missing values were handled using the SimpleImputer from scikit-learn, ensuring no gaps in the dataset. Multicollinearity was checked using the Variance Inflation Factor (VIF), and features with high VIF were removed.
- **Exploratory Data Analysis (EDA):** Various visualizations were used to identify trends and relationships between the features and the target variable.
- **Assumption Testing:** The assumptions of linear regression were validated through Linearity, Homoscedasticity, Normality of Residuals, and No Autocorrelation.

### 2.3. Tools Used
- **Programming Language:** Python
- **Libraries:** NumPy, Pandas, scikit-learn, Matplotlib, Seaborn, Statsmodels

## 3. Analysis and Results

### 3.1. Key Findings
Polynomial Regression (Degree 2) outperformed the other models with the lowest MAE (0.72), MSE(2.84), and RMSE (1.69). It also achieved the highest R2 Score (0.87), indicating that the model explained 87% of the variance in the target variable.

### 3.2. Visualizations
- **Scatter Plots:** Used to check the relationships between features and the target variable.
- **Residual Plots:** Visualized the spread of errors, confirming the homoscedasticity assumption.
- **Q-Q Plots:** Examined the normality of residuals.
- **Box Plots:** Identified potential outliers.

### 3.3. Model Performance
- **Linear Regression & Multiple Linear Regression:** MAE: 1.153, MSE: 3.288, RMSE: 1.813, R² Score: 0.848
- **Polynomial Regression (Degree 2):** MAE: 0.723, MSE: 2.843, RMSE: 1.686, R² Score: 0.869
- **Polynomial Regression (Degree 3):** MAE: 1.87, MSE: 36.69, RMSE: 6.06, R² Score: -0.69 (Overfitting issue)
- **Ridge & Lasso Regression:** Performance varied, with overfitting observed in some cases.

## 4. Actionable Insights

### 4.1. Feature Engineering
It is essential to explore additional feature transformations or interactions between variables, such as encoding categorical variables in more sophisticated ways, or using logarithmic transformations to handle skewed data.

### 4.2. Model Tuning
Although Polynomial Regression (Degree 2) performed well, experimenting with more complex models like Gradient Boosting or XGBoost might yield even better results, especially with feature interactions.

### 4.3. Regularization
Ridge and Lasso performed poorly in this task, likely due to over-regularization. Fine-tuning their hyperparameters or considering different regularization techniques might improve their performance.

## 5. Conclusion
This study successfully predicted car selling prices using regression models. Polynomial Regression (Degree 2) emerged as the best model based on performance metrics, striking a good balance between bias and variance. Although further refinement of the model is possible, the current approach provides a solid foundation for accurate car price predictions. Future work could involve feature engineering, experimenting with more advanced models, and further validation using real-world data.

---

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/AyaSaadawi/Econometrics-Price-Prediction.git
    ```

2. Navigate to the project directory:
    ```bash
    cd Econometrics-Price-Prediction
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the project, use the following command to execute the script:

```bash
python car_price_prediction.py
