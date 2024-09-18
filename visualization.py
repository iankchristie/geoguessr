from data_set import StreetViewDataset
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functions import *
import torch
from torchvision import transforms
import pdb


def visualize(image_tensor, true_lat, true_lon, target_values):
    # Display the image using Plotly
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Input Image",
            "Output Target: {:.2f}°, {:.2f}°".format(true_lat, true_lon),
        ),
        specs=[
            [{"type": "xy"}, {"type": "geo"}]
        ],  # Set first subplot to "xy", second to "geo"
    )
    add_input_image(fig, image_tensor)
    add_output_target(fig, target_values)
    fig.update_layout(
        width=1000,
        height=600,
    )
    fig.show()


def add_input_image(fig, image_tensor):
    # Unnormalize the tensor if it was normalized (e.g., with ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    image_tensor = image_tensor * std[:, None, None] + mean[:, None, None]

    # Convert the tensor to a NumPy array and permute to get (H, W, C)
    image_np = image_tensor.permute(1, 2, 0).numpy()

    # Clip values to ensure they are between 0 and 1
    image_np = np.clip(image_np, 0, 1)

    # Convert the image to 8-bit format (0-255) for display
    image_np = (image_np * 255).astype(np.uint8)

    fig.add_trace(go.Image(z=image_np), row=1, col=1)


def add_output_target(fig, target_values):
    num_points = len(target_values)

    _, _, _, lat, lon, _, _ = fibonacci_sphere(len(target_values))

    target_values_nums = [tensor.item() for tensor in target_values]

    hover_text = np.array(
        [
            "Value: {:2f}".format(
                target_values_nums[i],
            )
            for i in range(num_points)
        ]
    )

    fig.add_trace(
        go.Scattergeo(
            lat=lat,
            lon=lon,
            mode="markers",
            marker=dict(
                size=2,
                color=target_values_nums,
                colorscale="Viridis",
                colorbar=dict(title="Heatmap"),
                showscale=True,
            ),
            text=hover_text,
        ),
        row=1,
        col=2,
    )
    fig.update_geos(
        projection_type="equirectangular",
        row=1,
        col=2,
    )


if __name__ == "__main__":
    num_output_neurons = 10_000
    target_sigma = 0.1
    data_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to fit the input size of ResNet
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # Normalize based on ImageNet mean and std
        ]
    )

    _, _, _, grid_lat, grid_lon, _, _ = fibonacci_sphere(num_output_neurons)

    # Dataset and DataLoader
    dataset = StreetViewDataset(
        root_dir="data_2",
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        target_sigma=target_sigma,
        transform=data_transforms,
    )

    image_tensor, target, lat, lon = dataset.__getitem_with_lat_lon__(1)
    visualize(image_tensor, lat, lon, target)
