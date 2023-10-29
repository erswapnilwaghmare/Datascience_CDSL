## QCut(L1)
1. **What**: `qcut` is a function in pandas used for quantile-based discretization of continuous data.
2. **Why**: It helps divide data into specified quantiles, making it easier to analyze and categorize numeric values.
3. **How**: You provide the data, specify the number of quantiles, and optionally assign labels to bins.
4. **Use**: It's used for creating categorical variables from numerical data based on quantiles, useful for grouping and analysis.
5. **Without It**: You'd need to manually calculate quantile boundaries and create categories, which can be more complex and error-prone.

* Example:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Sample data: Exam scores and student names for 4 students
data = {'Student': ['Alice', 'Bob', 'Charlie', 'David'],
        'Exam_Score': [45, 62, 75, 80]}

# Create a DataFrame
df = pd.DataFrame(data)

# Manually compute quantiles (percentiles)
quantiles = [0, 0.25, 0.5, 0.75, 1.0]

# Use cut() to categorize scores into performance levels
df['Performance_Level'] = pd.cut(df['Exam_Score'],q=quantiles,labels=["Poor","Average","Execellent"])

# Display the result with student names and mapped performance levels
print(df)

```

* Output:

```
   Student  Exam_Score Performance_Level
0    Alice          45              Poor
1      Bob          62           Average
2  Charlie          75         Execellent
3    David          80         Execellent

```