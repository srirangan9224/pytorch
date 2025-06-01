import torch  # imports pytorch
import tqdm  # import tqdm for status bar
from torchvision import (
    datasets,
    transforms,
)  # import the datasets and transforms from torchvision
import matplotlib.pyplot as plt
import numpy as np

# split the dataset as 80 / 10 / 10 for train/ val / test
TRAIN_SIZE = 0.8
TEST_SIZE = 0.1
VAL_SIZE = 0.1
LEARNING_RATE = 0.001  # learning rate of the model
EPOCH = 25  # epoch size of 5 (loop over the dataset epoch times)
LABELS = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def test_image(img):
    """
    This function allows the user to check the images in the dataset

    Args:
        img (tuple): it is a tuple that stores the image tensor and the image label
    """
    x, y = img
    x = x / 2 + 0.5  # unnormalize the image
    plt.imshow(
        np.transpose(x.numpy(), [1, 2, 0])
    )  # change it so that we can display it on matplotlib
    print(f"the object is a {LABELS[y]}")  # print out the label


def evaluate(model, loader, loss_function):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_images = 0
    with torch.no_grad():  # test without gradients
        for image, label in loader:
            predicted_label = model(image)
            loss = loss_function(predicted_label, label)
            total_loss += loss.item() * image.size(0)
            total_correct += (predicted_label.argmax(1) == label).sum().item()
            total_images += image.size(0)
    loss = total_loss / total_images
    accuracy = total_correct / total_images
    return (loss, accuracy)


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer1 = torch.nn.Conv2d(
            3, 10, 5
        )  # create a convolutional layer with 3 input channels, 6 output channels and a 5x5 kernel matrix
        self.pool = torch.nn.MaxPool2d(2, 2)  # create a 2x2 pooling layer
        self.conv_layer2 = torch.nn.Conv2d(
            10, 16, 5
        )  # create another layer with different values
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 240)  # create linear layers
        self.fc2 = torch.nn.Linear(240, 120)
        self.fc3 = torch.nn.Linear(120, 60)
        self.fc4 = torch.nn.Linear(60, 10)
        self.dropout = torch.nn.Dropout(0.2)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        # define the forwarding conditions
        x = self.conv_layer1(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.conv_layer2(x)
        x = self.act(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.act(x)
        x = self.fc4(x)
        return x


# convert the given dataset into a tensor and normalize it
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)  # load the entire dataset
trainSet, valSet, testSet = (
    torch.utils.data.random_split(  # split the dataset into train test and val sets
        dataset, [TRAIN_SIZE, VAL_SIZE, TEST_SIZE]
    )
)

# create data loaders for the three data subsets
trainLoader = torch.utils.data.DataLoader(
    trainSet, batch_size=4, shuffle=True, num_workers=0
)
valLoader = torch.utils.data.DataLoader(
    valSet, batch_size=4, shuffle=False, num_workers=0
)
testLoader = torch.utils.data.DataLoader(
    testSet, batch_size=4, shuffle=False, num_workers=0
)

# create an instance of the model
model = CNN()

# define the loss and optimizer variables
loss_function = torch.nn.CrossEntropyLoss()  # loss function using Cross entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # adam optimizer
epoch_losses = []

# training the model
for epoch in tqdm.tqdm(range(EPOCH)):
    epoch_loss = 0
    for image, label in trainLoader:
        optimizer.zero_grad()  # clears the accumulated gradients
        predicted_label = model(image)  # get a prediction from the model
        loss = loss_function(
            predicted_label, label
        )  # calculate the loss and optimize the weights
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(trainLoader)
    epoch_losses.append(epoch_loss)
    print(f"Loss: {epoch_loss}")

# validation
validation_loss, validation_accuracy = evaluate(model, valLoader, loss_function)
print(f"Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}")

# testing
test_loss, test_accuracy = evaluate(model, testLoader, loss_function)
print(f"Testing Loss: {test_loss}, Testing Accuracy: {test_accuracy}")

example_input = torch.rand(1, 3, 32, 32)
scripted_model = torch.jit.trace(model, example_input)
scripted_model.save("model.pt")