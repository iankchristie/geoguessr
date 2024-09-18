from model import GeoLocalizationModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from functions import *
from visualization import *
from data_set import *
import pdb

if __name__ == "__main__":
    num_output_neurons = 10_000
    target_sigma = 0.1
    _, _, _, grid_lat, grid_lon, _, _ = fibonacci_sphere(num_output_neurons)

    data_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to fit the input size of ResNet
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # Normalize based on ImageNet mean and std
        ]
    )

    # Dataset and DataLoader
    dataset = StreetViewDataset(
        root_dir="data_2",
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        target_sigma=target_sigma,
        transform=data_transforms,
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoLocalizationModel(output_size=num_output_neurons).to(device)
    criterion = nn.MSELoss()  # Mean Squared Error for comparing distributions
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
        model.train()  # Set the model to training
        print(f"Running Epochs")
        for epoch in range(num_epochs):
            running_loss = 0.0
            batch_num = 0
            for inputs, targets in dataloader:
                print(f"Batch: {batch_num}")
                inputs, targets = inputs.to(device), targets.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                outputs = torch.softmax(
                    outputs, dim=1
                )  # Softmax to get probability distribution
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Accumulate loss
                running_loss += loss.item()
                batch_num = batch_num + 1

            # Print loss for every epoch
            epoch_loss = running_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    print(f"Starting Training")

    train_model(model, dataloader, criterion, optimizer, num_epochs=10)

    torch.save(model.state_dict(), "geo_localization_model_with_gaussian.pth")
