## Confusion matrix

* A confusion matrix is a table used to describe the performance of a classification model by showing the actual versus predicted values

## Syntax:

```
[[ True Positives (TP), False Negatives (FN) ],
 [ False Positives (FP), True Negatives (TN) ]]

```

## Code Examples

```python
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Sample employee data
data = {
    'EmployeeID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'Department': ['Sales', 'HR', 'IT', 'Sales', 'HR', 'IT', 'Sales', 'HR', 'IT', 'Sales'],
    'YearsAtCompany': [5, 2, 7, 10, 1, 3, 4, 8, 2, 5],
    'PerformanceRating': [3, 4, 2, 5, 3, 5, 3, 4, 4, 3],
    'ActualPromotion': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No'],
    'PredictedPromotion': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'No', 'No']
}

# Creating DataFrame from the sample data
df = pd.DataFrame(data)

# Confusion matrix calculation
conf_matrix = confusion_matrix(df['ActualPromotion'], df['PredictedPromotion'], labels=["Yes", "No"])

# Print the confusion matrix
print(conf_matrix)

# Plotting the confusion matrix as a heatmap
sns.set()  # Use seaborn's default settings
plt.figure(figsize=(8, 6))  # Specify figure size
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Predicted: Yes", "Predicted: No"], yticklabels=["Actual: Yes", "Actual: No"])
#'''
cm: Your confusion matrix data.
annot=True: This will annotate each cell in the heatmap with the numeric value from the confusion matrix.
fmt="d": This is the format for the annotations. In this case, it's set to display integers.
cmap="Blues": The colormap used for the heatmap. In this case, it's set to "Blues," which is a blue color scheme.
xticklabels: Labels for the x-axis ticks. In your case, it's set to ["Predicted: Yes", "Predicted: No"].
yticklabels: Labels for the y-axis ticks. In your case, it's set to ["Actual: Yes", "Actual: No"].
Remember to adjust the parameters based on your specific needs and the structure of your confusion matrix,,,

plt.xlabel('Predicted Promotion')
plt.ylabel('Actual Promotion')
plt.title('Confusion Matrix for Employee Promotions')
plt.show()

```

Output

```
[[3, 1],
 [0, 6]]

```

* Exaplantion:

* The model correctly predicted promotions for 3 employees (TP).
* The model correctly predicted non-promotions for 6 employees (TN).
* The model did not incorrectly predict any promotions for employees who did not get promoted (FP).
* The model missed 1 employee who got promoted, predicting that they would not get promoted (FN).
