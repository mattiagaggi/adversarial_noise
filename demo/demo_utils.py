from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import requests
import torch


def denormalize(tensor: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    """
    Denormalizes a tensor using the provided mean and standard deviation.

    Args:
        tensor (torch.Tensor): The input tensor to denormalize.
        mean (list): A list of mean values for each channel.
        std (list): A list of standard deviation values for each channel.

    Returns:
        torch.Tensor: The denormalized tensor.
    """
    mean = torch.tensor(mean, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, 3, 1, 1)
    tensor = tensor * std + mean
    return tensor


def imshow(tensor: torch.Tensor, title: str = None) -> None:
    """
    Displays an image from a tensor.

    Args:
        tensor (torch.Tensor): The tensor representing the image.
        title (str, optional): The title of the image. Defaults to None.

    Returns:
        None
    """
    image = tensor.cpu().clone().squeeze(0).detach()  # Detach tensor from the graph
    image = denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = image[0].permute(1, 2, 0)  # Rearrange dimensions to HWC
    image = image.clamp(0, 1)  # Ensure the values are in the range [0, 1]
    plt.imshow(image.numpy())
    if title:
        plt.title(title)
    plt.axis('off')


def download_image(url: str) -> Image.Image:
    """
    Downloads an image from a URL.

    Args:
        url (str): The URL of the image to download.

    Returns:
        Image.Image: The downloaded image.

    Raises:
        HTTPError: If the request to download the image fails.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        print("Image downloaded successfully")
        return Image.open(BytesIO(response.content))
    else:
        print(f"Failed to download image. Status code: {response.status_code}")
        response.raise_for_status()
