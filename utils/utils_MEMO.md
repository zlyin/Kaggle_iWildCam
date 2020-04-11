# CLAHE

# HOG features

# Background Subtraction





# Label Smoothing
- Visualization
	- traditional *Hard Label Assignment* `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]`
	- *Soft Label Assignment* `[0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.91 0.01 0.01]`
- Motivation
	- Softmax使用one-hot对真实标签进行编码;
	- 将标签强制one-hot的方式使网络过于自信会导致过拟合; need to soften it!
- Equation
	- ![img](https://pic3.zhimg.com/80/v2-0724ff964d48fe56dbc16e54c1691606_720w.png)
	- left = soften predicted distribution
	- right = the label has prob = \epsilon to come from uniform distribution;
		while has prob = (1 - \epsilon) to come from the original distribution;
	- Equivalent to adding noise to the original predicted labels to avoid
		concentrating on high probs classes;
- Cross Entropy
	- ![img](https://pic1.zhimg.com/80/v2-d12f119588a7e9ff5e73b61f1febf97c_720w.png)
	- new CE loss becomes = Loss of Prediction & Groundtruth distribution + Loss
		of Prediction & Prior distribution (uniform);
- Core
	- **Soften** the one-hot encoding prediction

