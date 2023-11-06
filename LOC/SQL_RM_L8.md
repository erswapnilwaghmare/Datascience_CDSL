## SQL_Roadmap 
| Category                | SQL Statement                                             | Description                                                                     | Tips                                                                            |
|-------------------------|-----------------------------------------------------------|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| **Basic Queries**       | `SELECT column1, column2 FROM table_name;`                | Retrieve specific columns from a table.                                         | Use `*` to select all columns.                                                  |
| **Filtering Data**      | `SELECT * FROM table_name WHERE condition;`               | Retrieve data that meets certain conditions.                                    | Combine conditions with `AND`, `OR`.                                           |
| **Sorting Results**     | `SELECT * FROM table_name ORDER BY column ASC/DESC;`      | Sort results by a specific column in ascending or descending order.             | Use `ASC` for ascending, `DESC` for descending order.                          |
| **Aggregate Functions** | `SELECT COUNT(*), SUM(column), AVG(column), MAX(column), MIN(column) FROM table_name;` | Use aggregate functions for statistical calculations over a set of rows. | Often used with `GROUP BY` for grouped statistics.                             |
| **Joining Tables**      | `SELECT * FROM table1 JOIN table2 ON table1.column = table2.column;` | Combine rows from two or more tables, based on a related column.           | Types of joins include `INNER JOIN`, `LEFT JOIN`, `RIGHT JOIN`, `FULL JOIN`.   |
| **Inserting Data**      | `INSERT INTO table_name (column1, column2) VALUES (value1, value2);` | Insert new data into a table.                                                   | Match the values with the columns in order.                                     |
| **Updating Data**       | `UPDATE table_name SET column1 = value1 WHERE condition;` | Update existing data within a table.                                            | Always specify a condition, or all records will be updated.                     |
| **Deleting Data**       | `DELETE FROM table_name WHERE condition;`                 | Delete data from a table.                                                       | Be cautious with the condition to avoid deleting unintended rows.               |
| **Creating Table**      | `CREATE TABLE new_table_name ( column1 datatype, column2 datatype, ... );` | Create a new table.                                                         | Define all columns and data types. Optional constraints can be added.           |
| **Altering Table**      | `ALTER TABLE table_name ADD new_column datatype;`         | Add a new column to an existing table.                                          | To remove a column, use `DROP COLUMN column_name;`.                             |
| **Dropping Table**      | `DROP TABLE table_name;`                                  | Delete a table and all of its data permanently.                                 | Use with caution - this action cannot be undone.                                |
| **Grouping Data**       | `SELECT column, COUNT(*) FROM table_name GROUP BY column;` | Group data in one column and perform aggregate functions on another.        | Often used with `HAVING` to filter groups.                                      |
| **Using Subqueries**    | `SELECT * FROM table_name WHERE column IN (SELECT column FROM another_table);` | Use nested SELECT statements to filter data based on another selection.    | Subqueries can be used in `SELECT`, `FROM`, `WHERE` clauses.                    |
| **Creating Indexes**    | `CREATE INDEX index_name ON table_name (column);`         | Create an index on a table column to improve the speed of data retrieval.       | Indexes can speed up searches but slow down data insertion and update.          |
| **Transaction Control** | `BEGIN TRANSACTION;`<br>`...SQL Statements...`<br>`COMMIT;`<br>or<br>`ROLLBACK;` | Manage transactions to ensure data integrity. | `COMMIT` saves all changes made in the transaction, `ROLLBACK` undoes them. |

Tips for Beginners:
- Practice with real data: Use sample databases like Northwind or AdventureWorks for practice.
- Be mindful of data types: Make sure your data types are correct when creating or altering tables.
- Always back up your data: Especially before performing operations that modify or delete data.
- Optimize your queries: Learn to write efficient queries to improve performance as you get more advanced.
- Use comments: Use `--` for single-line comments and `/* ... */` for multi-line comments to document complex queries.
- Study database normalization: Understanding normalization will help you design efficient databases.
