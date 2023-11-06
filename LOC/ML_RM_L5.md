## Machine_Learning_Roadmap

| Category                     | Command/Code                                                  | Description                                                                 |
|------------------------------|---------------------------------------------------------------|-----------------------------------------------------------------------------|
| **General Import**           | `import sklearn`                                              | Import the scikit-learn library                                            |
| **Dataset Split**            | `from sklearn.model_selection import train_test_split`        | Import train/test split function                                            |
|                              | `X_train, X_test, y_train, y_test = train_test_split(X, y)`   | Split the dataset into training and testing sets                            |
| **Data Preprocessing**       | `from sklearn.preprocessing import StandardScaler`            | Import StandardScaler for feature scaling                                   |
|                              | `scaler = StandardScaler()`                                  | Create a StandardScaler object                                              |
|                              | `X_train_scaled = scaler.fit_transform(X_train)`              | Fit to data, then transform it                                              |
|                              | `X_test_scaled = scaler.transform(X_test)`                    | Perform standardization by centering and scaling                            |
| **Model Training**           | `from sklearn.linear_model import LogisticRegression`         | Import Logistic Regression                                                  |
|                              | `model = LogisticRegression()`                               | Create a Logistic Regression classifier                                     |
|                              | `model.fit(X_train, y_train)`                                | Train the model using the training sets                                     |
| **Model Prediction**         | `predictions = model.predict(X_test)`                         | Predict the response for test dataset                                       |
| **Model Evaluation**         | `from sklearn import metrics`                                | Import scikit-learn metrics module                                          |
|                              | `metrics.accuracy_score(y_test, predictions)`                | Calculate the accuracy score                                                |
|                              | `metrics.confusion_matrix(y_test, predictions)`               | Compute confusion matrix to evaluate the accuracy of a classification       |
| **Cross-Validation**         | `from sklearn.model_selection import cross_val_score`        | Import cross-validation function                                            |
|                              | `scores = cross_val_score(model, X, y, cv=5)`                | Perform 5-fold cross-validation                                             |
| **Hyperparameter Tuning**    | `from sklearn.model_selection import GridSearchCV`           | Import GridSearchCV for hyperparameter tuning                               |
|                              | `parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}`      | Define parameter grid for tuning                                            |
|                              | `grid_search = GridSearchCV(estimator, parameters, cv=5)`    | Perform a grid search on the 'parameter grid' with cross-validation         |
| **Clustering**               | `from sklearn.cluster import KMeans`                         | Import KMeans for clustering                                                |
|                              | `kmeans = KMeans(n_clusters=3)`                              | Create a KMeans instance with 3 clusters                                    |
|                              | `kmeans.fit(X)`                                              | Compute k-means clustering                                                  |
| **Feature Selection**        | `from sklearn.feature_selection import SelectKBest`           | Import selection module for selecting features                              |
|                              | `selected_features = SelectKBest(k=2)`                       | Select the top 2 features                                                   |
| **Pipeline Creation**        | `from sklearn.pipeline import make_pipeline`                 | Import function to create a pipeline                                        |
|                              | `pipeline = make_pipeline(scaler, model)`                    | Create a pipeline with a scaler and an estimator                            |
| **Model Persistence**        | `from joblib import dump, load`                              | Import joblib for saving and loading models                                 |
|                              | `dump(model, 'model.joblib')`                                | Save a model to disk                                                        |
|                              | `model = load('model.joblib')`                               | Load a model from disk                                                      |
