## Employee promotions Prediciton
---

# Employee Promotion Eligibility Predictor

The following Python script aims to predict whether an employee is eligible for promotion based on performance metrics.

```python
def predict_promotion(performance_rating, years_at_company, projects_completed):
    """
    Predicts promotion eligibility based on performance metrics.
    
    Parameters:
    performance_rating (int): The performance rating on a scale from 1-5.
    years_at_company (int): The number of years the employee has been at the company.
    projects_completed (int): The number of projects completed in the last year.
    
    Returns:
    str: A message indicating the promotion eligibility.
    """
    if performance_rating >= 4 and years_at_company >= 3 and projects_completed >= 5:
        return "Eligible for promotion"
    elif performance_rating >= 3 and projects_completed >= 3:
        return "Consideration for promotion"
    else:
        return "Not currently eligible for promotion"

# Sample usage:
if __name__ == "__main__":
    rating = int(input("Enter your performance rating (1-5): "))
    years = int(input("Enter the number of years at the company: "))
    projects = int(input("Enter the number of projects completed last year: "))
    promotion_status = predict_promotion(rating, years, projects)
    print(f"Promotion status: {promotion_status}")
```

## 1. Decision Tree Model for Promotion Prediction

The following code creates a decision tree classifier to help in predicting employee promotions based on their performance data.

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Example dataset with employee performance metrics.
data = [
    {'performance_rating': 5, 'years_at_company': 4, 'projects_completed': 6, 'promoted': 1},
    {'performance_rating': 2, 'years_at_company': 2, 'projects_completed': 3, 'promoted': 0},
    # ... more data points ...
]

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Separate features and target variable
X = df.drop('promoted', axis=1)
y = df['promoted']

# Initialize and train the Decision Tree Classifier
tree_model = DecisionTreeClassifier()
tree_model.fit(X, y)
```

## 2. Interactive User Input for Promotion Prediction

Here's how you can collect user input to predict promotion eligibility using the trained decision tree model.

```python
def collect_employee_performance_data():
    """
    Collects the performance data of an employee.
    
    Returns:
    dict: A dictionary of performance data.
    """
    print("Please enter your performance data for promotion evaluation:")
    employee_data = {
        'performance_rating': int(input("Performance Rating (1-5): ")),
        'years_at_company': int(input("Years at Company: ")),
        'projects_completed': int(input("Projects Completed in the Last Year: "))
    }
    return employee_data

def evaluate_promotion_criteria(employee_data):
    """
    Evaluates the promotion criteria using a decision tree classifier.
    
    Parameters:
    employee_data (dict): The employee's performance data.
    """
    employee_data_frame = pd.DataFrame([employee_data])
    promotion_prediction = tree_model.predict(employee_data_frame)
    if promotion_prediction[0] == 1:
        print("According to the model, you are eligible for promotion.")
    else:
        print("According to the model, you are not currently eligible for promotion.")

# Running the performance data collection and evaluation.
if __name__ == "__main__":
    employee_performance_data = collect_employee_performance_data()
    evaluate_promotion_criteria(employee_performance_data)
```
