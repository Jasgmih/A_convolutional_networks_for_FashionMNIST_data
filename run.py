
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
from model import CNNModel
import torch.nn as nn
import matplotlib.pyplot as plt

image_size = 28

# Load the dataset
train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashinMNIST', 
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
)

test_set = torchvision.datasets.FashionMNIST(
    root = './data/FashinMNIST', 
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
)

# Changed the pic pixel values from range [0,255] to range [0,1]
preprocess = transforms.Compose([
    transforms.Lambda(lambda x: x.data.float()),
    transforms.Normalize(mean=[0], std=[255.]),
])
train_data = preprocess(train_set.data)
test_data = preprocess(test_set.data)

train_targets_data = torch.utils.data.TensorDataset(train_data,train_set.targets)
test_targets_data = torch.utils.data.TensorDataset(test_data,test_set.targets)
num_test_examples = len(test_targets_data)

batch_size=64
train_loader = torch.utils.data.DataLoader(train_targets_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_targets_data, batch_size=batch_size, shuffle=True)


lr = 0.001
model = CNNModel()

loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr)

epochs = 8 # len(train_loader) = 60000, batch_size = 64, each epoch contains 60000/64=940次循环, 共940*16=15000次循环
loss_list,iteration_list,accuracy_list = [], [], []
count = 0

for epoch in range(epochs):
    for i,  (images, labels) in enumerate(train_loader):
        imgs = Variable(images.view(min(batch_size,len(images)),1,image_size, image_size))
        labels = Variable(labels)

        optim.zero_grad()
        prediction = model(imgs)
        loss = loss_fn(prediction, labels)
        loss.backward()
        optim.step()
        count+= 1

        if count%50 == 0:
            accuracy_count = 0
            accuracy_previous = 0
            for i, (images, labels) in enumerate(test_loader):

                imgs = Variable(images.view(min(batch_size,len(images)),1,image_size, image_size))
                labels = Variable(labels)

                prediction = model(imgs)
                accuracy_count += (prediction.argmax(axis=1) == labels).sum()

            accuracy = accuracy_count.item() / float(num_test_examples) * 100
            accuracy_list.append(accuracy)
            loss_list.append(loss.item())
            iteration_list.append(count)

            # save the best model
            if accuracy > accuracy_previous:
                checkpoint = {'model': model,'state_dict': model.state_dict(), 'accuracy': accuracy, 'loss':loss.item()}
                torch.save(checkpoint,'./model/checkpoint.pth')
            accuracy_previous = accuracy

        if count%100 == 0:
            print('Iteration: {}, Loss: {}, Accuracy: {:4.2f}%'.format(count,loss.item(),accuracy))

        # count += 1

        

plt.subplot(211)
plt.plot(iteration_list, loss_list)
plt.subplot(212)
plt.plot(iteration_list, accuracy_list)
plt.show()











