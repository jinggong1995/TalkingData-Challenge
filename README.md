# TalkingData-Challenge
This is my Kaggle competition of CTR prediction(https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection)

The results shown here is the final version of models with optimal feature and optimal parameters and ensemble weights.


## models and scores

model1 LGBM(gbdt) with 27 (20 numerical, 7 categorical) features.

model2 LGBM(dart) with 27(17 numerical, 7 categorical) features.

model	private score: 0.9794777

model public score: 0.9793582

## feature engineering and importance testing

feature engineering can be found in FE.py

1.counting features

2.cumulative count

3.time to next click

4.time count

5.variance

6.common IP

7.unique count

Features will be calculated once and saved to disk.

Feature importance testing can be found in LightGBM.py

## Requirements
I used following environment

AWS EC2:

Memory: 256GB RAM, 256GB SWAP

CPU: 4 core

GPU: NVIDIA K80

Python3 packages:

numpy==1.14.2

pandas==0.22.0

lightgbm==2.1.0

tensorflow==1.5.1

## Reference

Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree". In Advances in Neural Information Processing Systems (NIPS), pp. 3149-3157. 2017.

Qi Meng, Guolin Ke, Taifeng Wang, Wei Chen, Qiwei Ye, Zhi-Ming Ma, Tieyan Liu. "A Communication-Efficient Parallel Algorithm for Decision Tree". Advances in Neural Information Processing Systems 29 (NIPS 2016).

Huan Zhang, Si Si and Cho-Jui Hsieh. "GPU Acceleration for Large-scale Tree Boosting". arXiv:1706.08359, 2017.

DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
