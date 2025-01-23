# West Bengal COVID-19 Dataset - Linear Regression Implementation  

This project is a visual and statistical implementation of Linear Regression using the COVID-19 dataset for West Bengal. The goal is to predict the number of confirmed cases on a particular day using Python libraries and machine learning techniques.  

## Project Overview  

- **Libraries Used**:  
  - `numpy` for numerical computations.  
  - `pandas` for data manipulation and analysis.  
  - `matplotlib` for data visualization.  
  - `scikit-learn` for machine learning.  
  - `datetime` for handling date conversions and manipulations.  

- **Objective**:  
  Predict confirmed COVID-19 cases using Linear Regression and evaluate its performance with metrics like MAE, MSE, and RMSE.  

## Steps Implemented  

### 1. Data Import and Visualization  
- Imported the COVID-19 dataset into Jupyter Notebook.  
- Read the dataset using `pandas` and plotted the **Date vs Confirmed Cases** graph using `matplotlib`.  
- Observed statistical details of the data for insights.  

### 2. Data Preparation  
- **Attributes and Labels**:  
  - **Attributes**: `Date` column (used as input to predict cases).  
  - **Labels**: `Confirmed Cases` column.  
- Used `iloc` (integer-location-based indexing) to prepare the dataset by selecting required columns.  

### 3. Data Splitting  
- Split the dataset into training and testing sets using Scikit-Learn's `train_test_split()`.  
- **Training Set**: 70% of the data.  
- **Test Set**: 30% of the data.  
- Specified the `test_size` parameter to define the test data proportion.  

### 4. Linear Regression Model  
- Trained the Linear Regression model using Scikit-Learn’s `LinearRegression` class.  
- Called the `fit()` method to train the model on the training dataset.  
- The model calculated the optimal intercept and slope for the best-fitting line.  

### 5. Model Evaluation  
Evaluated the model using the following metrics:  
1. **Mean Absolute Error (MAE)**: Mean of absolute errors.  
2. **Mean Squared Error (MSE)**: Mean of squared errors.  
3. **Root Mean Squared Error (RMSE)**: Square root of MSE.  

- Observed that the RMSE value was **1526**, which is around 50% of the mean value of confirmed cases (**2508.40**).  
- Concluded that the algorithm performed reasonably well on the dataset.  

### 6. Challenges and Solutions  
- **Date Conversion**: Converting date data into a numerical format was challenging. Used the `datetime` library to overcome this.  
- **Data Splitting**: Ensured proper splitting to maximize accuracy and efficiency.  

## Results and Insights  
- The Linear Regression model provided a decent performance in predicting confirmed cases.  
- RMSE of 1526 indicated a reasonably accurate prediction based on the dataset.  
- The visualization of **Date vs Confirmed Cases** offered valuable insights into the spread of COVID-19 in West Bengal.  

## Installation and Usage  

### Prerequisites  
Ensure the following Python libraries are installed:  
- numpy  
- pandas  
- matplotlib  
- scikit-learn  

