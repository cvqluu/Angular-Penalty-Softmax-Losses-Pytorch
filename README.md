# Angular Penalty Softmax Losses Pytorch
Concise Pytorch implementation of the Angular Penalty Softmax Losses presented in: 

* ArcFace: https://arxiv.org/abs/1801.07698 [1]
* SphereFace: https://arxiv.org/abs/1704.08063 [2]
* CosFace/Additive Margin: https://arxiv.org/abs/1801.09414 [3] / https://arxiv.org/abs/1801.05599 [4]

(Note: the SphereFace implementation is not exactly as described in their paper but instead uses the 'trick' presented in the ArcFace paper to use arccosine instead of the double angle formula)

```python

from loss_functions import AngularPenaltySMLoss

in_features = 512
out_features = 10 # Number of classes

criterion = AngularPenaltySMLoss(in_features, out_features, loss_type='arcface') # loss_type in ['arcface', 'sphereface', 'cosface']

# Forward method works similarly to nn.CrossEntropyLoss
# x of shape (batch_size, in_features), labels of shape (batch_size,)
# labels should indicate class of each sample, and should be an int, l satisying 0 <= l < out_dim
loss = criterion(x, labels) 
loss.backward()
```

## Experiments/Demo

There are a simple set of experiments on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) [2] included in `train_fMNIST.py` which compares the use of ordinary Softmax and Additive Margin Softmax loss functions by projecting embedding features onto a 3D sphere.

The experiments can be run like so:

``
python train_fMNIST.py --num-epochs 40 --seed 1234 --use-cuda
``

Which produces the following results:

### Baseline (softmax)
![softmax](figs/baseline.png?raw=true "softmax")

### Additive Margin Softmax/CosFace
![cosface](figs/cosface.png?raw=true "cosface")

### ArcFace
![arcface](figs/arcface.png?raw=true "arcface")

TODO: fix sphereface results

[1] Deng, J. et al. (2018) ‘ArcFace: Additive Angular Margin Loss for Deep Face Recognition’. Available at: http://arxiv.org/abs/1801.07698.

[2] Liu, W. et al. (2017) ‘SphereFace: Deep hypersphere embedding for face recognition’, in Proceedings - 30th IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, pp. 6738–6746. doi: 10.1109/CVPR.2017.713.

[3] Wang, H. et al. (2018) ‘CosFace: Large Margin Cosine Loss for Deep Face Recognition’. Available at: http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_CosFace_Large_Margin_CVPR_2018_paper.pdf (Accessed: 12 August 2019).

[4] “Additive Margin Softmax for Face Verification.” Wang, Feng, Jian Cheng, Weiyang Liu and Haijun Liu. IEEE Signal Processing Letters 25 (2018): 926-930.

[5] "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms." Han Xiao, Kashif Rasul, Roland Vollgraf. arXiv:1708.07747
