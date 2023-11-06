# BMI Calculator using Python

The following Python script calculates the Body Mass Index (BMI) and interprets the result according to the WHO standards.

```python
def calculate_bmi(weight_kg, height_cm):
    """
    Calculates the Body Mass Index (BMI) and returns the BMI category.
    
    Parameters:
    weight_kg (float): Weight of the individual in kilograms.
    height_cm (float): Height of the individual in centimeters.
    
    Returns:
    str: A message indicating the BMI category.
    """
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obesity"

# Sample usage of the BMI function:
if __name__ == "__main__":
    weight = float(input("Enter your weight in kilograms: "))
    height = float(input("Enter your height in centimeters: "))
    bmi_category = calculate_bmi(weight, height)
    print(f"Your BMI category is: {bmi_category}")
```

## 1. Naïve Bayes Model for Diabetes Risk Prediction

The following Python code snippet simulates the creation of a Naïve Bayes classifier to assess the risk of diabetes based on user-provided health data.

```python
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Simulated dataset with health metrics and diabetes outcome.
data = [
    {'glucose_level': 85, 'blood_pressure': 80, 'insulin': 0, 'bmi': 22, 'diabetes': 0},
    {'glucose_level': 168, 'blood_pressure': 74, 'insulin': 0, 'bmi': 35, 'diabetes': 1},
    # Additional records would be added here...
]

# Convert to DataFrame.
df = pd.DataFrame(data)

# Split data into features and target.
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Initialize the Gaussian Naive Bayes model.
model = GaussianNB()
model.fit(X, y)
```

## 2. Collecting User Health Data for Risk Prediction

The following code provides a way to interact with users to collect health data for diabetes risk prediction using the trained Naïve Bayes model.

```python
def collect_health_data():
    """
    Prompts the user to input health metrics for diabetes risk prediction.
    
    Returns:
    dict: A dictionary containing the user's health metrics.
    """
    print("Please enter the following health metrics:")
    health_data = {
        'glucose_level': float(input("Enter your glucose level (mg/dL): ")),
        'blood_pressure': float(input("Enter your blood pressure (mmHg): ")),
        'insulin': float(input("Enter your 2-Hour serum insulin (mu U/ml): ")),
        'bmi': float(input("Enter your Body Mass Index (BMI): "))
    }
    return health_data

def predict_diabetes_risk(health_metrics):
    """
    Predicts diabetes risk based on user's health metrics.
    
    Parameters:
    health_metrics (dict): Health metrics provided by the user.
    """
    health_metrics_frame = pd.DataFrame([health_metrics])
    prediction = model.predict(health_metrics_frame)
    if prediction[0] == 1:
        print("The model predicts a higher risk of diabetes. Please consult a healthcare professional.")
    else:
        print("The model predicts a lower risk of diabetes. Maintain a healthy lifestyle.")

# Example execution of health data collection and risk prediction.
if __name__ == "__main__":
    user_health_data = collect_health_data()
    predict_diabetes_risk(user_health_data)
```

