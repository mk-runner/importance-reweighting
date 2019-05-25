# importance-reweighting方法实现
## 参考文献：Liu T, Tao D. Classification with Noisy Labels by Importance Reweighting[J]。
## 一、方法实现说明
1.编程语言：python，需要numpy模块，以及KLIEP模块（见：https://github.com/Liuykang/KLIEP）
2.本文采用KLIEP方法估计条件概率，并且直接采用真实噪声率
3.训练集标签为{+1，-1}
## 二、使用指南
sample_weight = cal_sample_weigth(train, label_noise, sigma)
accuracy = kernel_SVM(train, test, sample_weight=sample_weight)
