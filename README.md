
# Student-Performance-Prediction

A machine learning project that analyzes student academic performance and predicts final exam scores using Python and Linear Regression. The project includes data cleaning, exploratory analysis, model training, evaluation, and visualization to examine the relationship between weekly study hours and final scores.

## Objectives
- Analyze student performance data.
- Identify and handle missing values.
- Explore the distribution of final exam scores.
- Predict final scores using Linear Regression.
- Evaluate model performance using regression metrics.
- Visualize actual scores and model predictions.

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Dataset
**File:** `students_performance_dataset.csv`

- Records: 3,000
- Features used: `Study_Hours_per_Week`
- Target variable: `Final_Score`

The model uses weekly study hours as the input feature to predict students' final exam scores.

## Project Workflow
1. **Data Loading:** Import and inspect the dataset using Pandas.
2. **Data Cleaning:** Check missing values and remove rows with missing study hours or final scores.
3. **Feature Selection:** Select study hours as X and final score as y.
4. **Train-Test Split:** Divide the data into 80% training and 20% testing sets.
5. **Model Training:** Train a Linear Regression model.
6. **Prediction:** Predict final scores on the test dataset.
7. **Evaluation:** Measure prediction performance using MAE, MSE, RMSE, and R².

## Visualizations

### 1. Distribution of Final Exam Scores
A histogram showing the frequency distribution of students' final exam scores.

### 2. Actual vs Predicted Scores
A scatter plot showing actual test scores and a regression line representing predicted scores based on weekly study hours.

## Model Evaluation
The model is evaluated using:

- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of Determination

These metrics help assess prediction errors and how well the model explains variation in final scores.

## Installation and Usage

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/Student-Performance-Prediction.git
```

### 2. Navigate to the Project
```bash
cd Student-Performance-Prediction
```

### 3. Install Dependencies
```bash
pip install numpy pandas matplotlib scikit-learn
```

### 4. Run the Script
Ensure the Python script and CSV dataset are in the same directory.

```bash
python students_performance.py
```

## Project Structure
```text
Student-Performance-Prediction/
│
├── students_performance.py
├── students_performance_dataset.csv
└── README.md
```

## Limitations
- The model uses only weekly study hours as a predictor.
- It assumes a linear relationship between study hours and final scores.
- The analysis identifies associations, not causal effects.

## Future Improvements
- Include additional academic features.
- Compare multiple regression algorithms.
- Improve model performance and evaluation.
- Explore the influence of attendance, assignments, and midterm scores.
