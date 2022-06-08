# A_convolutional_networks_for_FashionMNIST_data
This repository aims to process the FashinMINST dataset with a convolutional networks, and output the predication categories.
You could have a brief preview of the dataset by checking the check_dataset.ipynb script.

The convolutional network architecture is build in the model.py script, containing 4 convolutional layers, plus 2 layer normalization and 2 max pool layers as below.

CNNModel(
  (cnn1): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1))
  (relu1): ReLU()
  (layernorm1): LayerNorm((24, 24), eps=1e-05, elementwise_affine=True)
  (cnn2): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1))
  (relu2): ReLU()
  (layernorm2): LayerNorm((20, 20), eps=1e-05, elementwise_affine=True)
  (cnn3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (relu3): ReLU()
  (pool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (cnn4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
  (relu4): ReLU()
  (pool4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (linear): Linear(in_features=576, out_features=10, bias=True)
)

To train the model simply run the run.py script, the model will be saved to 'model/checkpoint.pth' at the end.
After training, you could load the pre-trained model and check the predication results in Predication.ipynb script.

