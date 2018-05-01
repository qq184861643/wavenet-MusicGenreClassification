# wavenet-MusicGenreClassification
## Mapping music genres with wavenet
### Purpose
Testify the capacity of wavenet in multi-labeling task of sequential data.
### Baseline and Dataset
Baseline: [A Deep Learning Approch for Mapping Music Genres](https://ieeexplore.ieee.org/document/7994970/). 

Test AUC score:89.4, Training time on one Nvidia 1080Ti: about 20 hours.

Dataset: MagnaTagATune

### Model Structure
A cascade wavenet works in parallel with a CNN. The wavenet with M dilation depth and N residual blocks output (batch_size,64) tensor, while CNN output tensors of the same size. Concatenate them or plus them together and through fc layers the output the predictions.
Details are in the code.

### What have I done
BATCH_SIZE=16, EPOCH=23.<br>

**1**.Using only cascade wavenet with 6 dilation depth 5 residual blocks can reach around 84.66 AUC score and only 7000 seconds training time. Indicate that wavenet is capable of modeling the music spectrogram and much faster than RNN structure.

**2**. wavenet: 6 dilation depth, 5 residual blocks and 64-d outputs;

CNN:basically all 128 conv kernels with 64-d outputs;

Results: test AUC around 88.13, training time around 14000 seconds.


**3**.wavenet: 8 dilation depth, 4 residual blocks and 64-d outputs;

CNN:basically all 128 conv kernels with 64-d outputs;

Results: test AUC around 88.07, training time around 14000 seconds. Indicate that increasing the receptive field of wavenet cannot improve the model performance.

**4**.wavenet: 6 dilation depth, 5 residual blocks and 128-d outputs;

CNN:basically all 128 conv kernels with 128-d outputs;

Results: test AUC around 88.57, training time around 14000 seconds.

**5**.model structure is the same as in 4.

As the 'False' label is far more than 'True' label in training dataset's label, I've added a weight term in binary_cross_entropy loss as below:<br>
```python
def loss(self,outputs,target,alpha = 1.1):
		x = outputs
		y = target
		epsilon = 1e-5
		one_to_zero = -torch.log(x+epsilon)*y
		zero_to_one = -torch.log(1-x+epsilon)*(1-y)
		loss = torch.mean(torch.mean(one_to_zero*alpha+zero_to_one,dim=1))
		return loss
```
I find the test AUC score raise rapidly to 87.91 at around 13 epoch but reach 88.54 after training is done. So the weight term accelerates the convergence but does not help with final performance.

**6**. I tried to increase the conv kernels of the parallel CNN but the model overfitted.<br>
**7**. I tried to change the gate function from sigmoid to elu inside wavenet's residual block but led to a worse result.<br>
**8**. I removed all the Dropout layers entangled with BatchNorm layers to avoid [Variance Shift](https://arxiv.org/abs/1801.05134). With some minor structure tunings the model reach 88.71 test auc score and it needs only 10 epochs(about 0.49 seconds a step, 6000 seconds in total) to converge. With SEnet struture the auc score decrease to 88.6 while converging a little faster.<br>
**9**. Tried ResNeXt block structure on CNN, resulting a slightly pooler test auc score(88.46).

### Conclusion of my work
The model reaches a relativly good result(88.71 compared to baseline 89.4) while massively reduces the training time(about 6000 seconds compared to about 20 hours), proving that the causal 1-d dilated convolution network has quite potential in sequential data classification, which could to be exploited in future works(i.e. Adding attention in wavenet structure).
