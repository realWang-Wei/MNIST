MNIST
===
# Introduction
MNIST is a classic data set in the field of machine learning and deep learning. There are many methods for this data set. Here are some methods for reference.



# Algorithms
* ML
  * SVM
  * DecesionTree
  * RandomForest
  * KNeighbors
  * Adaboost
  * XGBoost
  * catboost
  * lgbm
  
* DL
  * FC
  * CNN（VGG16）
  * LSTM
  * BLS

# Score on test data（kaggle）
here is [kaggle Digit Recongizer](https://www.kaggle.com/c/digit-recognizer)

The accuracy is verified in the kaggle competition, and all algorithms have not been adjusted or optimized.



Test on: GPU: Tesla P100 x1
         CPU: 8 kernels, 64G RAM
         
|Algorithm|Score|Time Cost for training/s|
|--|--|--|
|SVM| 0.11614| 8504|
|DecisionTree|0.85585 |10.90 |
|RandomForest| 0.94142 | 2.61|
|KNeighbors|0.96800 |5.63 |
| Adaboost |0.72914 |23.32 |
|XGBoost| | |
|catboost | | |
|lgbm| | |
|neural network with numpy |0.92214 |662 (10 epochs)|
|VGG16|0.98828| 676 (32 epochs)|
|LSTM| 0.90785|1447 (32 epochs)|
|BLS|0.93471 |30.43|
