import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def get_dominant_colors(image_path, n_colors=2):
    """
    Extracts the dominant colors from an image using KMeans clustering.

    Parameters:
        image_path (str): Path to the image file.
        n_colors (int): Number of dominant colors to extract.

    Returns:
        np.ndarray: An array of the dominant colors in RGB format.
    """

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)

    # Use KMeans clustering to find the dominant colors
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)

    # Get the colors and their respective counts
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    label_counts = np.bincount(labels)

    # Sort the colors by the number of pixels
    sorted_indices = np.argsort(label_counts)[::-1]
    sorted_colors = colors[sorted_indices]

    return sorted_colors


def rgb_to_hex(rgb):
    """Converts RGB color to hexadecimal"""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def plot_colors(colors, image_name):
    """
    Plots the dominant colors and saves the figure.

    Parameters:
        colors (np.ndarray): An array of dominant colors.
        image_name (str): The base name of the image for saving the plot.
    """
    plt.figure(figsize=(8, 3))
    plt.axis("off")

    for idx, color in enumerate(colors):
        hex_color = rgb_to_hex(color)
        plt.fill_between([idx, idx + 1], 0, 1, color=hex_color)
        plt.text(
            idx + 0.5,
            0.5,
            hex_color,
            fontsize=12,
            ha="center",
            va="center",
            color="white" if np.mean(color) < 128 else "black",
        )

    plt.xlim(0, len(colors))
    plt.ylim(0, 1)
    plt.title("Dominant Colors", fontsize=16)

    plt.savefig(f"color_palette_{image_name}.png", bbox_inches="tight")
    plt.close()


def main():
    # Main function
    imgs_folder = "imgs"
    if not os.path.isdir(imgs_folder):
        raise FileNotFoundError(f"Folder: '{imgs_folder}' does not exist.")

    image_files = [
        f
        for f in os.listdir(imgs_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        raise ValueError("No image files found in the specified folder.")

    for image_file in image_files:
        image_path = os.path.join(imgs_folder, image_file)
        dominant_colors = get_dominant_colors(image_path, n_colors=2)
        plot_colors(dominant_colors, os.path.splitext(image_file)[0])


if __name__ == "__main__":
    main()
