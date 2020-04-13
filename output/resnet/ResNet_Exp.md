# EXP1
- Data
	- fetched 24 classes and 501 images in total
	- class weights (min, max) = (1.0, 1.105263157894737)
	- training data has 400 samples over 24 classes
	- validation data has 101 samples over 24 classes
- LR
	|	Epoch		|	Learning Rate	|	Val Acc 	|
	|---------------------------------------------------|
	|	1->130		| 	1e-1			|	0.8317		|
	|	130->170	| 	1e-1			|	0.8317		|
- Test score
	- @Stage3:
```
                           precision    recall  f1-score   support
                 accuracy                           0.91       101
                macro avg       0.91      0.92      0.90       101
             weighted avg       0.95      0.91      0.92       101
```
- Obviously overfitting, acc reaches 1;

# EXP2
- Data
	- add more categories that less than 1k images
	- fetched 168 classes and 21012 images in total
	- class weights (min, max) = (1.0, 51.73684210526316)
	- training data has 16809 samples over 168 classes
	- validation data has 4203 samples over 168 classes
- LR
	|	Epoch		|	Learning Rate	|	Val Acc 	|
	|---------------------------------------------------|
	|	1->40		| 	1e-1			|	0.58		|
	|	41->55 		| 	1e-2			|			|
- Test score
	- @Stage1:
		```
                          precision    recall  f1-score   support
                 accuracy                           0.58      4203
                macro avg       0.40      0.52      0.40      4203
             weighted avg       0.68      0.58      0.60      4203
		```
- learning curve plateaus, learning stagnates;

# EXP3
- Data
	- fetched 209 classes and 108764 images in total
 	- class weights (min, max) = (1.0, 678.3157894736842)
 	- training data has 87011 samples over 209 classes
 	- validation data has 21753 samples over 209 classes
- LR
	|	Epoch		|	Learning Rate	|	Val Acc 	|
	|---------------------------------------------------|
	|	1->5		| 	1e-1			|	0.5052		|
	|	6->15 		| 	1e-2			|				|
- Test score
	- @Stage1:



# EXP4 - 2GPUs
- Data
	- fetched 209 classes and 108764 images in total
 	- class weights (min, max) = (1.0, 678.3157894736842)
 	- training data has 87011 samples over 209 classes
 	- validation data has 21753 samples over 209 classes
- Model:
	- still use the `resnet dedicated to "cifar10"`
- LR
	|	Epoch		|	Learning Rate	|	Val Acc 	|
	|---------------------------------------------------|
	|	1->12		| 	1e-1			|	0.6466		|
	|	13->25		| 	1e-2 			| 	0.84		|
	|	26->30		| 	1e-3 			| 	0.90		|

- Test score
	- @Stage3:
		```
                         precision    recall  f1-score   support
                 accuracy                           0.90     21753
                macro avg       0.68      0.81      0.71     21753
             weighted avg       0.91      0.90      0.90     21753
			 ```
- Sbumission
	- @Epoch 35 => 0.624
	

# EXP5 - 2GPUs
- Data
	- **create HDF5 files** 
		- fetched 209 classes and 108764 images in total
		- class weights (min, max) = (1.0, 687)
		- training data has 87011 samples over 209 classes
		- validation data has 21753 samples over 209 classes
	- `dataset forgot to / 255!`
- Model
	- New `ResNet50` with `Conv7x7 => Act => BN => MaxPool2D` in the 1st block
- LR
	|	Epoch		|	Learning Rate	|	Val Acc 	|
	|---------------------------------------------------|
	|	1->15		| 	1e-1			|				|
	|	16->30		| 	1e-2 			| 				|
	|	31->50		| 	1e-3 			| 				|
	|	51->60		| 	1e-4 			| 0.86			|
- Test score
	- @Stage4:
			```
                           precision    recall  f1-score   support
                 accuracy                           0.86     21753
                macro avg       0.59      0.71      0.61     21753
             weighted avg       0.88      0.86      0.87     21753
			 ```
- Sbumission
	- @Epoch 60 => 0.651


# EXP6 - 2GPUs
- Data
	- exactly the same data HDF5 files as EXP5 
- Model
	- still use the `resnet dedicated to "cifar10"`
- LR
	|	Epoch		|	Learning Rate	|	Val Acc 	|
	|---------------------------------------------------|
	|	1->15		| 	1e-1			|	0.61		|
	|	16->30		| 	1e-2 			| 	0.82		|
	|	31->50		| 	1e-3 			| 	0.90		|
	|	51->60		| 	1e-4 			| 	0.90		|

- Test score
	- @Stage4:
		```
					   precision    recall  f1-score   support
			 accuracy                           0.90     21753
			macro avg       0.64      0.76      0.67     21753
		 weighted avg       0.91      0.90      0.90     21753
		```
- Submission
	- *@Epoch 60 => 0.693*


# EXP7 - 2GPUs
- Data
	- exactly the same data HDF5 files as EXP5 & 6
- Model
	- still use the `resnet dedicated to "cifar10"`
- Preprocessing 
	- rebuild HDF5 dataset to add **Label Smoothing to One-Hot encoded labels** 
	- `dataset.db["labels"]` is changed!

- `Label smoothing factor = 0.2`
	- use `LRDecayScheduler` to reduce lr automatically

		|	Epoch		|	Learning Rate	|macro f1_score |
		|---------------------------------------------------|
		|	1->40		|1e-1, power=2.5	| 	0.63 		|

	- Test score:
		- @Epoch 40 
		```
					   precision    recall  f1-score   support
			 accuracy                           0.91     21753
			macro avg       0.61      0.72      0.63     21753
		 weighted avg       0.92      0.91      0.91     21753
		 ```
	- Submission 
		- => @Epoch 40 => 0.695

- change `Label smoothing factor = 0.1`
	- `dataset.db["labels"]` is changed!
	- use `LRDecayScheduler` to reduce lr automatically
		|	Epoch		|	Learning Rate	|macro f1_score |
		|---------------------------------------------------|
		|	1->40		|1e-1, power=2.5	| 0.68			|
	- Test score:
		- @Epoch 40
		```
						precision    recall  f1-score   support
			 accuracy                           0.91     21753
			macro avg       0.66      0.77      0.68     21753
		 weighted avg       0.92      0.91      0.91     21753
		```
	- Submission
		- ***@Epoch 40 => 0.702***

- change `Label smoothing factor = 0.01`
	- `dataset.db["labels"]` is changed!
	- use `LRDecayScheduler` to reduce lr automatically

		|	Epoch		|	Learning Rate	|macro f1_score |
		|---------------------------------------------------|
		|	1->40		|1e-1, power=2.5	| 0.72			|

	- Test score:
		- @Epoch 40
			```
                           precision    recall  f1-score   support
                 accuracy                           0.91     21753
                macro avg       0.70      0.79      0.72     21753
             weighted avg       0.92      0.91      0.91     21753
			```
	- Submission:
		- @Epoch 40 => 0.697
		- *Kind of overfitting??*
- Conclusion ==> **LB uses macro f1_score**; `LB score = local macro f1_score + 0.02`


# EXP8 
- Based on Exp 7
	- exactly the same data HDF5 files as EXP5 
	- still use the `resnet dedicated to "cifar10"`
- Preprocessing 
	- add **label_moothing=0.1 to CategoricalCrossentropy directly** 
	- use `LRDecayScheduler` to reduce lr automatically 
- LR
	|	Epoch		|	Learning Rate	|	Val Acc 	|
	|---------------------------------------------------|
	|	1->40		| 	1e-4 			| 0.65			|

- Test score
	- @Stage4: 	**need to modify preprocessors for predict_v2.py!**
		```
                          	precision    recall  f1-score   support
				 accuracy                           0.91     21753
                macro avg       0.63      0.75      0.65     21753
             weighted avg       0.92      0.91      0.91     21753
		```
- Submission
	- **@Epoch 40 => 0.703**
	- **Apply label smoothing to loss function is a little bit better!**


# EXP9
- Based on Exp 5 & 6
	- exactly the same data HDF5 files as EXP5 & 6
	- still use the `resnet dedicated to "cifar10"`
- **Difference**
	- replace the routine `weights based on sample frequency` with `weights
		based on effective number of classes`
	- add `LRScheduler` for automatic lr reduction
- LR
	|	Epoch		|	Learning Rate	|macro f1_score	|
	|---------------------------------------------------|
	|	1->40		| 	1e-1			|	0.74		|

- Test score
	- @Stage4:
		```
                           precision    recall  f1-score   support
                 accuracy                           0.91     21753
                macro avg       0.71      0.82      0.74     21753
             weighted avg       0.92      0.91      0.91     21753
		```
- Submission
	- @Epoch 40 => 0.698



# EXP10 
- Data
	- exactly the same data HDF5 files as EXP5 & 6 
- Model
	- still use the `resnet dedicated to "cifar10"`
- Preprocessing 
	- add **CLAHE and  WhiteBalance** to training & validation data
	- add `LRScheduler` for automatic lr reduction
- LR
	|	Epoch		|	Learning Rate	|macro f1_score	|
	|---------------------------------------------------|
	|	1->40		| 	1e-1			|0.68			|

- Try 1
	- Test score
		- @Stage4:
			```
							precision    recall  f1-score   support

					 accuracy                           0.89     21753
					macro avg       0.66      0.74      0.68     21753
				 weighted avg       0.89      0.89      0.89     21753
			```
	- Submission
		- @Epoch 40 => **need to resubmit!**
		- **using CLAHE + WhiteBalance together with `mean subtraction` will RUIN the image!**

- Try2 = Retrain, remove mean subtraction and not apply `1/255`!
	- Test score
		- @Stage4:
				```
								precision    recall  f1-score   support
					 accuracy                           0.89     21753
					macro avg       0.67      0.77      0.70     21753
				 weighted avg       0.90      0.89      0.89     21753
				```
	- Submission
		- @Epoch 40 => 0.695
		- **using CLAHE + WhiteBalance for augmentation, using `1 / 255` for normalization**


# EXP11
- Based on Exp 8
	- exactly the same data HDF5 files as EXP8 (with smoothing_factor=0.1) 
	- still use the `resnet dedicated to "cifar10"`
	- add `label_moothing=0.1 to CategoricalCrossentropy directly`
	- use `LRDecayScheduler` to reduce lr automatically 
- **Difference/Add-on**
	- replace the routine `weights based on sample frequency` with `weights
		based on effective number of classes`
- LR
	|	Epoch		|	Learning Rate	|	Val Acc 	|
	|---------------------------------------------------|
	|	1->40		| 	1e-4 			| running|

- Test score
	- @Stage4: 	**need to modify preprocessors for predict_v2.py!**
		```
    	```
- Submission
	- @Epoch 40 =>

