## Importing Data in Pandas

* Pandas library to import data from various sources and formats. 

1. **CSV Files**:
   You can read data from CSV (Comma Separated Values) files using the `pd.read_csv()` function.

   ```python
   import pandas as pd
   df = pd.read_csv('data.csv')
   ```

   You can specify various options like delimiter, encoding, and header row using function arguments.

2. **Excel Files**:
   To read data from Excel files, you can use the `pd.read_excel()` function.

   ```python
   import pandas as pd
   df = pd.read_excel('data.xlsx')
   ```

   You may need to install the `openpyxl` or `xlrd` library depending on the format of your Excel file.

3. **SQL Databases**:
   Pandas can also read data from SQL databases using the `pd.read_sql()` function. You'll need to establish a connection to the database using a library like SQLAlchemy or pyodbc.

   ```python
   import pandas as pd
   from sqlalchemy import create_engine

   engine = create_engine('sqlite:///mydatabase.db')
   df = pd.read_sql('SELECT * FROM mytable', engine)
   ```

4. **JSON Data**:
   You can import data from JSON (JavaScript Object Notation) files or URLs using `pd.read_json()`.

   ```python
   import pandas as pd
   df = pd.read_json('data.json')
   ```

5. **HTML Tables**:
   You can scrape HTML tables from web pages using the `pd.read_html()` function. It returns a list of DataFrame objects, one for each table found on the page.

   ```python
   import pandas as pd
   tables = pd.read_html('https://example.com')
   df = tables[0]  # Assuming you want the first table
   ```

6. **Clipboard**:
   You can read data from your clipboard using `pd.read_clipboard()`. This is useful when you have data copied to your clipboard from a source like Excel.

   ```python
   import pandas as pd
   df = pd.read_clipboard()
   ```

7. **Other Formats**:
   Pandas supports many other formats like HDF5, Parquet, and more. You can use functions like `pd.read_hdf()`, `pd.read_parquet()`, etc., to read data from these formats.

8. **From Lists**:
   You can create a DataFrame from a list of lists, where each inner list represents a row of data.

   ```python
   import pandas as pd

   data = [['Alice', 25], ['Bob', 30], ['Charlie', 35]]
   columns = ['Name', 'Age']
   df = pd.DataFrame(data, columns=columns)
   ```

9. **From a Dictionary**:
   You can create a DataFrame from a dictionary where keys are column names and values are lists of data for each column.

   ```python
   import pandas as pd

   data = {'Name': ['Alice', 'Bob', 'Charlie'],
           'Age': [25, 30, 35]}
   df = pd.DataFrame(data)
   ```

10. **From a NumPy Array**:

   You can create a DataFrame from a NumPy array, which is useful for numerical data.

   ```python
   import pandas as pd
   import numpy as np

   data = np.array([[1, 2], [3, 4]])
   df = pd.DataFrame(data, columns=['A', 'B'])
   ```

11. **From a List of Dictionaries**:
   You can create a DataFrame from a list of dictionaries, where each dictionary represents a row of data.

   ```python
   import pandas as pd

   data = [{'Name': 'Alice', 'Age': 25},
           {'Name': 'Bob', 'Age': 30},
           {'Name': 'Charlie', 'Age': 35}]
   df = pd.DataFrame(data)
   ```

12. **Adding Rows and Columns**:
   You can also create an empty DataFrame and then add rows and columns one by one.

   ```python
   import pandas as pd

   df = pd.DataFrame(columns=['Name', 'Age'])
   df = df.append({'Name': 'Alice', 'Age': 25}, ignore_index=True)
   df = df.append({'Name': 'Bob', 'Age': 30}, ignore_index=True)
   ```
