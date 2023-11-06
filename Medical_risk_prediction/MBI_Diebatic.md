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


## Fever Test using Python

The following Python script can be used to determine if an individual might have a fever based on their body temperature in Celsius.

```python
def check_for_fever(temperature_celsius):
    """
    Determines whether the input temperature is indicative of a fever.
    
    Parameters:
    temperature_celsius (float): Body temperature in degrees Celsius.
    
    Returns:
    str: A message indicating the presence of a fever or not.
    """
    # Fever threshold is set at 38.0 degrees Celsius.
    fever_threshold = 38.0
    
    # Check if the temperature is equal to or above the threshold.
    if temperature_celsius >= fever_threshold:
        return "You may have a fever."
    else:
        return "You likely do not have a fever."

# Demonstration of function usage:
if __name__ == "__main__":
    temperature_input = float(input("Enter your body temperature in Celsius: "))
    print(check_for_fever(temperature_input))
```

## 1. Logistic Regression Model for Predicting Typhoid

To simulate a classification model that predicts typhoid fever from symptoms, the Python code snippet below uses `pandas` for data manipulation and `sklearn.linear_model` for logistic regression.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

# A simulated dataset representing individuals with and without typhoid.
data = [
    # Example cases with typhoid (typhoid = 1)
    {'prolonged_fever': 1, 'headache': 1, 'fatigue': 1, 'abdominal_pain': 1, 'diarrhea': 1, 'rash': 0, 'typhoid': 1},
    {'prolonged_fever': 1, 'headache': 1, 'fatigue': 1, 'abdominal_pain': 1, 'diarrhea': 0, 'rash': 1, 'typhoid': 1},
    # Example cases without typhoid (typhoid = 0)
    {'prolonged_fever': 0, 'headache': 0, 'fatigue': 0, 'abdominal_pain': 0, 'diarrhea': 0, 'rash': 0, 'typhoid': 0},
    {'prolonged_fever': 0, 'headache': 0, 'fatigue': 0, 'abdominal_pain': 0, 'diarrhea': 1, 'rash': 0, 'typhoid': 0},
    # Additional examples would include a mix of symptoms and outcomes.
]

# Creating a DataFrame from the simulated data.
df = pd.DataFrame(data)

# Isolating the features (symptoms) from the target variable (typhoid diagnosis).
X = df.drop('typhoid', axis=1)
y = df['typhoid']

# Initializing the Logistic Regression model and training it with the dataset.
model = LogisticRegression()
model.fit(X, y)
```

## 2. Collecting User Input for Symptoms

For user interaction and prediction of typhoid based on symptoms, the following code provides a function to collect user input and another to make predictions using the trained logistic regression model.

```python
def collect_user_symptoms():
    """
    Asks the user to input their symptoms and returns a dictionary of these symptoms.
    
    Returns:
    dict: A dictionary with symptom presence (1) or absence (0).
    """
    print("Please enter your symptoms:")
    symptoms = {
        'prolonged_fever': int(input("Prolonged fever (more than 5 days)? (1 for Yes, 0 for No): ")),
        'headache': int(input("Headache? (1 for Yes, 0 for No): ")),
        'fatigue': int(input("Fatigue? (1 for Yes, 0 for No): ")),
        'abdominal_pain': int(input("Abdominal pain? (1 for Yes, 0 for No): ")),
        'diarrhea': int(input("Diarrhea? (1 for Yes, 0 for No): ")),
        'rash': int(input("Rash? (1 for Yes, 0 for No): "))
    }
    return symptoms

def make_typhoid_prediction(symptoms):
    """
    Predicts the likelihood of typhoid based on user-entered symptoms.
    
    Parameters:
    symptoms (dict): A dictionary of the user's symptoms.
    """
    symptoms_frame = pd.DataFrame([symptoms])
    prediction = model.predict(symptoms_frame)
    if prediction[0] == 1:
        print("The model predicts a possibility

 of typhoid. Please consult a doctor for a proper diagnosis.")
    else:
        print("The model does not suggest typhoid. However, if you feel unwell, please see a healthcare professional.")

# Executing the functions for demonstration.
if __name__ == "__main__":
    user_symptoms = collect_user_symptoms()
    make_typhoid_prediction(user_symptoms)
```

Note: The above model is for demonstration purposes only and should not be used as a substitute for professional medical advice. Always consult a healthcare provider for accurate diagnosis and treatment.


