# Additive Margin Softmax Loss Pytorch
Pytorch implementation of the Additive Margin Softmax Loss presented in https://arxiv.org/pdf/1801.05599.pdf [1]

```python

from AdMSLoss import AdMSoftmaxLoss

in_features = 512
out_features = 10 # Number of classes

criterion = AdMSoftmaxLoss(in_features, out_features, s=30.0, m=0.4) # Default values recommended by [1]

# Forward method works similarly to nn.CrossEntropyLoss
# x of shape (batch_size, in_features), labels of shape (batch_size,)
# labels should indicated class of each sample
loss = criterion(x, labels) 
loss.backward()
```

[1] Wang, Feng, Jian Cheng, Weiyang Liu and Haijun Liu. “Additive Margin Softmax for Face Verification.” IEEE Signal Processing Letters 25 (2018): 926-930.
