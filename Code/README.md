# Code Folder Instructions

## 1. data_process.py

This Python file should be implemented at first to collect data from [Kaggle](https://www.kaggle.com/c/whale-categorization-playground/data), and to generate cleaned training data and the oversampled data CSV files. To collect data, you should have a kaggle account with the kaggle.json at first. To obtain your API credentials, go to the [Account tab of your user profile](https://www.kaggle.com/me/account) and select Create API Token. Fill your username and key in the environment setting part of data_process.py, then you can download the data.

## 2. Train.py

This Python file should be secondly implemented, which contains cleaned training and validation data loading and processing, steps of network setting and training, and the best model saving.  

## 3. Test.py

This Python file should be implemented at last. It contains steps of test and validation data loading and processing, network loading, model testing (mean average precision computing), label predictions, and submit file generation.

##### The sources of codes I borrowed and adjusted from are listed in the project report.
