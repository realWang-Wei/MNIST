MNIST
===
# Introduction
MNIST is a classic data set in the field of machine learning and deep learning. There are many methods for this data set. Here are some methods for reference.

(MNIST是机器学习和深度学习领域一个经典的数据集，对于这个数据集有很多种方法，这里提供一些方法供参考。)

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

(准确率在kaggle的比赛上进行验证，所有算法都未经调参、优化等操作。)

Test on: GPU: Tesla P100 x1
         CPU: 8 kernels, 64G RAM
         
|Algorithm|Score|Time Cost for training/s|
|--|--|--|
|SVM| | |
|DecisionTree| | |
|RandomForest|  | |
|KNeighbors| | |
| Adaboost | | |
|XGBoost| | |
|catboost | | |
|lgbm| | |
|neural network with numpy |0.92214 |662 (10 epochs)|
|VGG16|0.98828| 676 (32 epochs)|
|LSTM| 0.90785|1447 (32 epochs)|
|BLS|0.93471 |30.43|
