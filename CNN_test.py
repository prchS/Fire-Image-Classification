import torch, torch.nn as nn
import kagglehub
from torchvision import transforms
from DataSet_class import FireDataset
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class FireCNN(nn.Module):
    def __init__(self):
        super(FireCNN, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Sequential(
            nn.LazyLinear(out_features=1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':

    root = kagglehub.dataset_download("dani215/fire-dataset") + "\\fire_dataset/"
    criterion = torch.nn.BCELoss()
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    seed = 42
    torch.manual_seed(seed)

    # Batch size - increased for milestone 3
    batch_size = 200
    # Must be a float between 0 and 1
    train_size = 0.7 # Train size
    val_size = 0.15 # Validation size
    test_size = 0.15 # Test size
    # Learning rate - increased for milestone 3
    lr = 0.002
    # Number of epochs - lowered for milestone 3
    num_epochs = 100

    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        # Adding transformations to the images for milestone 3 to reduce overfitting
        transforms.RandomGrayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomAffine(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    fire_dataset = FireDataset(root_dir=root, transform=transformations)
    train_dataset, test_dataset, validation_dataset = random_split(fire_dataset, [train_size, test_size, val_size])
    print("Test size:", len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Load the model
    model = torch.load('fire_cnn.pth')
    def test(model: nn.Module, test_loader: DataLoader, criterion: nn.Module):
        model.eval()
        metrics = {
            "test_loss": 0,
            "test_acc": 0,
            "test_f1": 0,
            "test_confusion_matrix": []
        }
        total_loss = 0
        total_acc = 0
        total_f1 = 0
        with torch.no_grad():
            for batch, (images, labels) in enumerate(test_loader):
                # Move images and labels to the device
                images, labels = images.to(device), labels.to(device)
                # Forward pass
                outputs = model(images)
                # Calculate the loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                # Get the predicted labels
                predicted = torch.round(outputs.cpu().detach())
                # Calculate the accuracy per batch
                total_acc += accuracy_score(labels.cpu(), predicted)
                # Calculate the F1 score per batch
                total_f1 += f1_score(labels.cpu(), predicted)
                # Calculate the confusion matrix only for the first batch
                if batch == 0:
                    metrics["test_confusion_matrix"].append(confusion_matrix(labels.cpu(), predicted))

            # Calculate the average metrics
            metrics["test_loss"] = total_loss / len(test_loader)
            metrics["test_acc"] = total_acc / len(test_loader)
            metrics["test_f1"] = total_f1 / len(test_loader)

            print(f"Test metrics: {metrics}")
        return metrics

    test(model, test_loader, criterion)