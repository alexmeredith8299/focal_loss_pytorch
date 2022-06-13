[![Documentation Status](https://readthedocs.org/projects/focal-loss-pytorch/badge/?version=latest)](https://focal-loss-pytorch.readthedocs.io/en/latest/?badge=latest)
# focal-loss-pytorch
Simple vectorized PyTorch implementation of binary unweighted focal loss as specified by [[1]](https://arxiv.org/pdf/1708.02002.pdf).

## Installation
This package can be installed using [pip](https://pip.pypa.io/en/stable/) as follows:
```python 
python3 -m pip install focal-loss-pytorch
```

## Example Usage
Here is a quick example of how to import the BinaryFocalLoss class and use it to train a model:
```python
from focal_loss_pytorch.focal_loss_pytorch.focal_loss import BinaryFocalLoss
import torch

#Initialize device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Initialize loss fn +  optimizer 
loss_fn = BinaryFocalLoss(gamma=5)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Load datasets
train_loader = DataLoader(train_set, batch_size=10, shuffle=False)
val_loader = DataLoader(val_set, batch_size=10, shuffle=False)

#Train! :)
for e in range(epochs):
   for data in train_loader:
      model.train()
      input_img = data['img'].to(device)
      ref_img = data['ref'].to(device)
      output_img = model(input_img)
            
     loss = loss_fn(output_img, ref_img)
     optimizer.zero_grad()
     loss.backward()
     optimizer.step()
```

## Documentation
Documentation for this package is available on [Read the Docs](https://focal-loss-pytorch.readthedocs.io/en/latest/).

## References
[1] Lin, T. Y., et al. "Focal loss for dense object detection." arXiv 2017." arXiv preprint arXiv:1708.02002 (2002).
