import torch,ssl
import torch.nn as nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader

ssl._create_default_https_context = ssl._create_unverified_context
transform=transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

train_data=datasets.CIFAR10(root="./data",train=True,transform=transform,download=True)
test_data=datasets.CIFAR10(root="./data",train=False,transform=transform,download=True)
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=1000)

model=models.resnet18(pretrained=True)
model.fc=nn.Linear(model.fc.in_features,10)

for param in model.parameters():
    param.requires_grad=False
for param in model.fc.parameters():
    param.requires_grad=True

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.fc.parameters(),lr=0.001)


model.train()
for epoch in range(5):
    for images,labels in train_loader:
        outputs=model(images)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss is {loss.item():.4f}")


correct=0
total=0
model.eval()
with torch.no_grad():
    for images,labels in test_loader:
        outputs=model(images)
        _,predictions=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predictions==labels).sum().item()
    print(f"correctness is {((correct/total)*100):.2f}")