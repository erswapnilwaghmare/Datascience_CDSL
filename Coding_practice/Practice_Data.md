* DataSet:

```
EmployeeID,Name,Age,Department,Salary,JoinDate,Gender,SeniorityLevel
1,John Doe,29,Marketing,55000,2018-06-12,Male,Senior
2,Jane Smith,34,IT,72000,2019-07-23,Female,Mid
3,Emily Davis,27,Operations,50000,2021-04-18,Female,Junior
4,Chris Morales,45,IT,95000,2015-02-15,Male,Senior
5,Martin Brown,22,Marketing,48000,2022-03-31,Male,Junior
6,Sophia Wilson,30,Finance,62000,2020-11-01,Female,Mid
7,Liam Johnson,41,Human Resources,68000,2016-12-04,Male,Senior
8,Olivia Lee,36,Finance,78000,2018-08-19,Female,Senior
9,William Davis,39,Operations,73000,2017-05-21,Male,Senior
10,Mia Rodriguez,25,IT,51000,2021-07-07,Female,Junior

```

### Basic Data Handling and Exploration

1. How do you load a CSV file into a Pandas DataFrame?
2. Display the first 10 rows of the dataset.
3. How many rows and columns are in the dataset?
4. List the names of all columns in the dataset.
5. What are the data types of each column?
6. Are there any missing values in the dataset? If so, in which columns?
7. Provide a statistical summary for numerical columns.
8. How can you change the column name 'JoinDate' to 'StartDate'?
9. How do you filter the rows that have missing 'Salary' information?
10. How can you retrieve a list of unique departments from the `Department` column?

### Data Cleaning

11. How would you fill missing values in the `Age` column with the average age?
12. How do you drop rows with any missing values?
13. Identify duplicate entries in the dataset. How would you handle them?
14. How do you convert the `JoinDate` from a string to a datetime data type?
15. If `SeniorityLevel` is categorical with 'Junior', 'Mid', 'Senior', how do you encode it as an ordinal variable?

### Data Analysis and Aggregation

16. What is the average salary across the entire dataset?
17. How many employees are there in each department?
18. What is the average age of employees in each `SeniorityLevel` category?
19. Which department has the highest average salary?
20. How do you calculate the median salary for male vs. female employees?
21. Display the average salary and age per department.
22. Find the top 3 departments with the most employees.
23. Calculate the sum of salaries in each department.
24. What is the min and max salary in the organization?
25. How many employees joined the company in the year 2022?

### Data Manipulation

26. How can you increase every employee's salary by 5%?
27. How do you group records based on gender and calculate the mean salary?
28. How would you extract the month from the `JoinDate` column?
29. How do you create a new column that shows years of experience based on `JoinDate`?
30. How can you create a pivot table showing the average salary by department and gender?

### Data Visualization

31. Can you create a histogram to show the distribution of salaries?
32. How would you plot a bar chart for the number of employees in each department?
33. Can you create a boxplot to compare the age distributions across different seniority levels?
34. How do you plot a line graph showing the trend of total salaries over the years?
35. Create a scatter plot to show the relationship between age and salary.

### Data Transformation

36. If `Salary` is in different currencies for different employees, how do you standardize it to USD?
37. How would you normalize the `Age` column?
38. How do you create a flag column indicating whether an employee's salary is above the average salary?
39. How do you apply a logarithmic transformation to the `Salary` column?
40. How can you convert categorical data in the `Department` column to dummy variables?

### Advanced Queries

41. How do you identify the top 10% highest-earning employees?
42. Can you find employees who have the word "Manager" in their job title within the dataset?
43. How do you rank employees based on their salary within each department?
44. How can you calculate the interquartile range for employee salaries?
45. Can you calculate the correlation coefficient between age and salary?

### Data Exporting

46. How do you export a DataFrame to a CSV file?
47. How can you save a DataFrame with only 'EmployeeID' and 'Salary' columns to a new CSV file?
48. What method would you use to export a DataFrame to an Excel file?
49. How do you write a DataFrame to a JSON file?
50. If you wanted to send the salary information to a SQL database, how would you do it?
