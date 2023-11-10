
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def show_images_in_row(image_list, titles=None, cmap=['tab20c']):
    # Calculate the number of images and create a single-row figure
    num_images = len(image_list)
    fig, ax = plt.subplots(1, num_images, figsize=(15, 5))
    if type(cmap) == str:
        cmap = [cmap] * num_images
    # Loop through the images and titles (if provided) to display them
    for i in range(num_images):
        ax[i].imshow(image_list[i], cmap=cmap[i])
        ax[i].axis('off')  # Turn off axis labels
        if titles:
            ax[i].set_title(titles[i])
    return fig

def mean_neighboring_integers_within_distance(int_list, max_distance):
    if len(int_list) < 2:
        return []  # There are not enough elements to compute neighboring means.

    neighboring_means = []
    
    for i in range(1, len(int_list)):
        distance = abs(int_list[i] - int_list[i - 1])
        if distance <= max_distance:
            mean = (int_list[i - 1] + int_list[i]) / 2.0
            neighboring_means.append(mean)

    return neighboring_means


# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes, num_in_channels):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_in_channels, out_channels=200, kernel_size=3)
        self.conv2 = nn.Conv2d(200, 100, 3)
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = self.conv2(x)
        x = nn.LeakyReLU()(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = nn.LeakyReLU()(x)
        x = self.fc2(x)
        return x
    






def show_images_in_row(image_list, titles=None, cmap=['tab20c']):
    # Calculate the number of images and create a single-row figure
    num_images = len(image_list)
    fig, ax = plt.subplots(1, num_images, figsize=(15, 5))
    if type(cmap) == str:
        cmap = [cmap] * num_images
    # Loop through the images and titles (if provided) to display them
    for i in range(num_images):
        ax[i].imshow(image_list[i], cmap=cmap[i])
        ax[i].axis('off')  # Turn off axis labels
        if titles:
            ax[i].set_title(titles[i])
    return fig

def overlay_images(grayscale_image, rgb_image, alpha=0.5):
    """
    Overlay a grayscale image on top of an RGB image with the specified alpha transparency.

    Parameters:
    - grayscale_image: The grayscale image to be overlaid.
    - rgb_image: The RGB image on which the grayscale image will be overlaid.
    - alpha: Transparency level (default is 0.5).

    Returns:
    - Blended image.
    """
    # Convert the grayscale image to an RGB image.
    
    grayscale_image_rgb = np.dstack((grayscale_image, grayscale_image, grayscale_image))

    # Blend the two images with the specified transparency.
    blended_image = alpha * grayscale_image_rgb + (1 - alpha) * rgb_image

    return blended_image

def label_map_to_rgb(label_map, cmap_name='tab20c'):
    # Create a colormap using the specified name (default is 'tab20c').
    cmap = plt.get_cmap(cmap_name)
    
    # Normalize label values to the range [0, 1].
    norm = plt.Normalize(vmin=0, vmax=np.max(label_map))
    
    # Apply the colormap to the label map and convert to RGB.
    label_map_rgb = cmap(norm(label_map))
    print(label_map.shape, label_map_rgb.shape)
    label_map_rgb[:, :, 2] = 0
    return label_map_rgb[:, :, :3]