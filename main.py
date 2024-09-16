import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import CNN
from func import train_step, test_step, accuracy

download = True

train_data = datasets.MNIST(
    root="image",
    train=True,
    download= download,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="image",
    train=False,
    download=download,
    transform=ToTensor()
)

# img, labels = train_data[0]
# plt.imshow(img.permute(1,2,0))
# plt.title(labels)
# plt.show()

BATCH_SIZE = 32
train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_cnn = CNN(input_shape=1,output_shape=10)
model_cnn.to(device)

cost_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_cnn.parameters(), lr=0.01)

EPOCHS = 10
for epoch in range(EPOCHS):
    print(f"Epoch: {epoch}\n----")
    train_step(train_dataloader, model_cnn, cost_fn, optimizer, accuracy, device)
    test_step(train_dataloader, model_cnn, cost_fn, accuracy, device)

torch.save(model_cnn.state_dict(), "MNIST_cnn.pth")




