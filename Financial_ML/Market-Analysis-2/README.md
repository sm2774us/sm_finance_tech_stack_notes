# Market-Analysis-2

### Overview

In Market Analysis 2 we upgrade Market Analysis to be better in  all of its component , plus that we add a new component called Feature-Calculator.

## Components

* [Dataset](#Dataset)

* [Feature-Calculator](#Feature-Calculator)

* [ML-Models](#ML-Models)

* [Tester](#Tester)

* [Connection](#Connection)

**Note:** The order in setup is important.

## Dataset

### Introduction

Create a Dataset for any symbol in any period of time in Forex market (Metatrader 4) that contain the basic Features
(open, high, low, close).

### How we do it
We do it by pulling data from MQL4 in to CSV file , the data is pulled using MQL4 build in functions that create our Features.

### Setup
1. Download the code in Dataset [(here)](https://github.com/Financial-ML/Market-Analysis-2/tree/master/DataSet)
1. Git in the code and write the number of bars you need.
1. Name the csv file then compile it.
1. Run the script (Dataset) in any symbol and any period of time.
#### Features
"OPEN","HIGH","LOW","CLOSE","VOLUME"
      open=OPEN(i);
      high=HIGH(i);
      low=LOW(i);
      close=CLOSE(i);
      volume=VOLUME(i);
   
## Feature-Calculator
### Introduction
In Feature-Calculator we calculate the machine learning Feature based on this research [(here)](http://www.wseas.us/e-library/conferences/2011/Penang/ACRE/ACRE-05.pdf).

### How we do it
* We build it in python .
* It based on the main Features [Dataset](#Dataset).
* And we save it after finsh extraction the feature in csv file.

### Setup
1. After calculating the [Dataset](#Dataset). copy the CSV file in to your python project.
1. Download the code in Feature-Calculators [(here)](https://github.com/Financial-ML/Market-Analysis-2/tree/master/Feature-Calculator) in to your python project.
1. Run the collector program and it will generate the Calculated Feature in a CSV file.



## ML-Models

### Introduction
Different Machine Learning models that we used to learn from the [Feature](#Feature-Calculator).

### How we do it
* We build the the models in python using scikit-learn.
* It learn from our predefined [Feature](#Feature-Calculator).
* And then save it after finsh traning in PKL file.

### Models
* Decision Tree.
* k-nearest neighbor.
* Logistic Regression.
* RandomForest.
* Support vector machine.
* Neural-network-MLPClassifier.

### Setup
1. After calculating the [Feature](#Feature-Calculator). copy the CSV file in to your python project.
1. Download the code in ML-Models [(here)](https://github.com/Financial-ML/Market-Analysis-2/tree/master/ML-Models) in to your python project.
1. Run the program and it will generate the PKL file.

## Tester

### Introduction
Tool that use to test the strategy that has been developed outside MQL4 in python.

### How we do it
* We do it by build our algorithmic trading strategy in python.
* And load our Models in Tester program.
* Then calculate the profits by saving the enter price then subtract from it the close price.

### Algorithmic Trading
We call the models (Decision Tree and RandomForest) every one hour and say if the first 10 min is the same prediction do the predicted action.
### What we test
1. Profit
1. Total number of trades
1. Sum of wining trades
1. Sum of loss trades
1. Max drawdown
1. Best trade

### Setup
1. After traning the models [ML-Models](#ML-Models).
1. Get your testing data from [Dataset](#Dataset).
1. Download the code in Tester [(here)](https://github.com/Financial-ML/Market-Analysis-2/tree/master/Tester) in to your python project.
1. Run the program and it will test the stratgy.

## Connection
### Introduction
Connect the strategy that has been developed in Python with fxcm company in a real time connection the structure has been developed to run on cloud every 1 hour.
### How we do it
* Pull the basic data from fxcm by connect to a generated token you can git it from [(here)](https://www.fxcm.com/markets/).
* After pulling data we predict the price using our machine learning models.
* All of this process is done in the cloud every 1 hour. 
### Setup
1. Create a new Python project.
1. Download the code in the project [(here)](https://github.com/Financial-ML/Market-Analysis-2/tree/master/Connector).
1. Copy the generated models from [ML-Models](#ML-Models).
1. Put it in a any cloud you are familiar with it.
1. If you use ubuntu use crontab to set a time to run the code.
1. the timer that are used in crontab is (11 * * * * * ) it means run the code every hour in minute 11.





