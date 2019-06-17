# Additive Margin Softmax Loss Pytorch
Pytorch implementation of the Additive Margin Softmax Loss presented in https://arxiv.org/pdf/1801.05599.pdf [1]

```python

from AdMSLoss import AdMSoftmaxLoss

in_features = 512
out_features = 10 # Number of classes

criterion = AdMSoftmaxLoss(in_features, out_features, s=30.0, m=0.4) # Default values recommended by [1]

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

### Additive Margin Softmax
![AdMSoftmax](figs/AdMSoftmax.png?raw=true "AdMSoftmax")

[1] “Additive Margin Softmax for Face Verification.” Wang, Feng, Jian Cheng, Weiyang Liu and Haijun Liu. IEEE Signal Processing Letters 25 (2018): 926-930.

[2] "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms." Han Xiao, Kashif Rasul, Roland Vollgraf. arXiv:1708.07747
