# Housing in Mexico City

## Purpose
The real estate market is a crucial aspect of any economy and understanding the factors that impact property prices can be of immense value to both buyers and sellers. The purpose of this project is to create a predictive model that estimates the prices of apartments in Mexico City based on various features such as surface covered, location, and borough. This project is focused on predictive data science and aims to provide insights into the real estate market in Mexico City.

## Data
The data used in this project was obtained from open sources and contains information on properties in Mexico City. The data was split into 5 CSV files and was pre-processed to remove any missing or irrelevant information. The main columns of interest after the cleaning process include:

* Price
* Surface covered
* Latitude and longitude
* Boroughs
The cleaning process was extensively documented in the project notebooks to ensure transparency and reproducibility.

## Methods
Two notebooks were used for this project, the first one was used for the initial walkthrough and wrangling, while the second was for modeling and deployment.
The following methods were used in this project:

* Ridge Regression: Ridge Regression is a type of linear regression that adds a penalty term to the loss function to prevent overfitting. Overfitting occurs when a model learns the noise in the data instead of the underlying patterns, leading to poor generalization performance on new data. Ridge Regression adds a penalty term that discourages large coefficients and helps to prevent overfitting.

* OneHotEncoder: This is a feature preprocessing technique that converts categorical variables into a binary representation suitable for use in a machine learning model. Categorical variables are variables that can take on one of a limited number of values. In this project, the borough column was converted into a binary representation using OneHotEncoder.

* SimpleImputer: This is a feature preprocessing technique that replaces missing values with a specified statistic (e.g. mean, median, etc.). Missing values can be problematic for machine learning models as they reduce the amount of data available for learning. SimpleImputer replaces missing values with either the mean or median value of the column, depending on the specified strategy.


## Results
The results of this project were communicated as follows:

* A visualization of the most important boroughs in Mexico City based on the model's predictions. This visualization provides insights into which boroughs have the highest and lowest predicted apartment prices.

* A function (make_prediction) that takes 4 arguments (surface covered, latitude, longitude, and borough) and returns the model's prediction for an apartment price. This function can be used to make predictions for new apartments based on their features.

* An interactive widget made with Jupyter widgets, which can be adjusted to see how predicted apartment prices change based on different input values. This widget provides an interactive way to explore the model's predictions and understand the impact of various features on apartment prices.

## Usage
It is important to note that this project is for educational purposes only and the results should not be used for real-world decision making without proper verification and validation.

## Conclusion
In conclusion, this project aimed to create a predictive model for apartment prices in Mexico City using data science techniques. The results of the project provide insights into the real estate market in Mexico City and can be used to make informed decisions about property investments. This project serves as an example of how data science can be applied to real-world
