# Bank Business Volume Time Series Forecast

## Introduction
In order to efficently allocation the sources of bank, it is very important for bank to predict the future business volumes.The task of this competition is to predict the future business volumes from **day granularity(sub task 1)** & **hour granularity(sub task 2)**.

## Data
the overall data structure is simple,just in form of **date -- period in the day -- volume**. 

There is another table to tell us the type of each day, for example, workday, weekend, hoilday and so on.

What needs additional attention is that there are two types of businesses,**A** and **B**,we need to deal with them respectively.

The training data includes the data from *2018.1.1* to *2020.10.31*, and our prediction targets are the business volumes for the day between *2020.11.1* to *2020.12.31*.

## Basic Solution

### 1.Preprocess and EDA
>Load data from csv and make some plot to observe the overall tendencies

### 2.Feature engineering
Besides the basic features the data provides like the date(year,month,day,hour,day type etc.),I also try to create some other features I think may help.

For example:

*some dummies*:
  
  i.**whether the previous day is weekend**
  
  ii.**whether the previous day is hoilday**

*information of the previous days*:

  iii.**the proportion of hoildays of the last 7 days**

  iv.**the proportion of workday-to-rest of the last 14 days**

  and so on.

### 3. Modeling
I try to use 3 types of modeling methods.

>1.The first is traditional time series forecast method like **ARIMA**, unfortunately, the performance of the model on validation set is not good.

>2.The second is **deep learning** method,LSTM.The performance of it on validation set is also not good,the possible reason is the data size is not large enough to fully give play to the power of deep learining.

>3.The final method is traditional **machine learning** method, I use 3 different **ensemble learning** methods.

>***---Random Forest*** ***---GBDT*** ***---Xgboost***


Together with the features I made before, these 3 models' performance on validation set far beyond the traditional time series method and deep learning methods.

Finally,I use the combination of these 3 models to predict the data on test set. And my final score(MAPE) on task 1 & 2 is:**0.136** & **0.7913** respectively.

### 4.Performance on validation set
>Task 1 *where the red line means the real values and the blue line means the predicted values*.

***Random Forest***

![random_forest](https://github.com/frankhjh/Fintech_TS_Forecast/blob/main/img/Random_Forest.png)

***GBDT***

![GBDT](https://github.com/frankhjh/Fintech_TS_Forecast/blob/main/img/GBDT.png)

***Xgboost***

![xgboost](https://github.com/frankhjh/Fintech_TS_Forecast/blob/main/img/Xgboost1.png)

>Task2 

***Xgboost***

![xgboost2](https://github.com/frankhjh/Fintech_TS_Forecast/blob/main/img/Xgboost2.png)



