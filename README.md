# EDS4 Final Project 2018
## Good Stock Recommendation by using Machine Learning

## Objective

Long term investment is one of the major investment strategies. However, calculating intrinsic value of some company and evaluating shares for long term investment is not easy, since analyst have to care about a large number of financial indicators and evaluate them in a right manner.
    
Our goal here is to identify good stocks that are growing healthily at a sustainable pace by using Machine Learning techniques by analyzing their key financial statistic ratio. With this, it is hopefully to reduce investor's effort and time to make tremendous study on annaul and/or quaterly report of each stock.

## Requirements

- Download all the Jupyter code files and  folders listed in Github to your local machine.
- Unzip "Forward_Finance_Data_Web_Files.zip" to folder "Forward_Finance_Data_Web_Files".
- Unzip "Processed_Historical_Finance_Data.zip" to folder "Processed_Historical_Finance_Data".
- Unzip content of 2 zip files in folder "Historical_Finance_Data" without creating new folder. After done, delete the zip files from folder.
- Get c0redumb/yahoo_quote_download code file from https://github.com/c0redumb/yahoo_quote_download (a copy of c0redumb code was attached). This code will help to acquired stock price for S&P 500 company.

## Jupyter Notebook

The Jupyter Notebook files should be read following sequence as per below:
- Good_Stock_Recommendation_by_using_Machine_Learning
- Good_Stock_Recommendation_by_using_Machine_Learning_Logistic_Regression
- Good_Stock_Recommendation_by_using_Machine_Learning_SVM
- Good_Stock_Recommendation_by_using_Machine_Learning_RandomForest
- Good_Stock_Recommendation_by_using_Machine_Learning_Others_Method


## Installation

2 installation needed for this project:

```shell

# To install Quandl's python package, use the package manager pip inside your terminal. For detail information, refer to https://www.quandl.com/
$ pip install quandl

# To install Bokeh which is a Python interactive visualization library that targets modern web browsers for presentation.
$ pip install bokeh

```

## Execute Stock Prediction via Plot

```shell

# With Anaconda Prompt, navigate to the plot folder, enter following command to show the web visualization
bokeh serve --show Stock.py

```
