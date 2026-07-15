# ResNet-18 Transfer Learning on CIFAR-10 using PyTorch

## Overview

This project demonstrates **Transfer Learning** using a **pretrained ResNet-18** model on the **CIFAR-10** dataset with PyTorch.

Instead of training the entire network from scratch, the pretrained ResNet-18 model is used as a feature extractor. All pretrained layers are frozen, and only the final fully connected (`fc`) layer is trained to classify the 10 classes of CIFAR-10.

---

## Features

- Uses pretrained **ResNet-18** from `torchvision.models`
- Transfer Learning approach
- Freezes pretrained convolutional layers
- Trains only the final classification layer
- Image resizing to **224 × 224**
- ImageNet normalization
- Evaluates classification accuracy on the test dataset

---

## Technologies Used

- Python
- PyTorch
- Torchvision

---

## Dataset

**CIFAR-10**

The dataset contains **60,000** color images of size **32 × 32** divided into **10 classes**.

Classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Dataset Split:

- Training Images: 50,000
- Test Images: 10,000

---

## Image Preprocessing

The images are transformed before being fed into ResNet-18.

```python
transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])
```

### Why Resize?

ResNet-18 is originally trained on ImageNet where input images are **224 × 224**.

### Why Normalize?

The normalization values correspond to the mean and standard deviation of the ImageNet dataset, ensuring compatibility with pretrained weights.

---

## Model Architecture

Pretrained Model:

```
ResNet-18
```

Modified Final Layer:

```python
model.fc = nn.Linear(model.fc.in_features, 10)
```

The original classifier predicts **1000 ImageNet classes**.

It is replaced with a new classifier that predicts **10 CIFAR-10 classes**.

---

## Transfer Learning

### Freeze all pretrained layers

```python
for param in model.parameters():
    param.requires_grad = False
```

### Unfreeze only the final layer

```python
for param in model.fc.parameters():
    param.requires_grad = True
```

This allows only the final classifier to learn while keeping the pretrained feature extractor unchanged.

---

## Loss Function

Cross Entropy Loss

```python
criterion = nn.CrossEntropyLoss()
```

---

## Optimizer

Adam Optimizer

```python
optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.001
)
```

Only the parameters of the final fully connected layer are optimized.

---

## Training

For each epoch:

1. Load a batch of images
2. Perform forward propagation
3. Compute loss
4. Clear previous gradients
5. Perform backpropagation
6. Update only the final layer weights

---

## Evaluation

During testing:

- Model switched to evaluation mode
- Gradient computation disabled
- Predictions generated
- Accuracy calculated

Accuracy formula:

```
Accuracy = (Correct Predictions / Total Images) × 100
```

---

## Project Structure

```
.
├── data/
├── resnet18_transfer_learning.py
└── README.md
```

---

## How to Run

Install dependencies:

```bash
pip install torch torchvision
```

Run the program:

```bash
python resnet18_transfer_learning.py
```

---

## Expected Output

```
Epoch 1/5, Loss: ...
Epoch 2/5, Loss: ...
Epoch 3/5, Loss: ...
Epoch 4/5, Loss: ...
Epoch 5/5, Loss: ...

Accuracy: XX.XX%
```

---

## Concepts Covered

- Transfer Learning
- Pretrained Models
- ResNet-18
- Feature Extraction
- Image Classification
- CIFAR-10
- PyTorch
- DataLoader
- Image Transformations
- CrossEntropyLoss
- Adam Optimizer
- Model Evaluation

---

## Future Improvements

- Fine-tune deeper ResNet layers
- Add learning rate scheduler
- Save and load trained model
- Plot training loss and accuracy
- Display confusion matrix
- Use GPU (CUDA) for faster training
- Train for more epochs to improve accuracy

---

## License

This project is intended for educational and learning purposes.
