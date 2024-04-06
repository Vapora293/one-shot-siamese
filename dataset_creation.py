import os
import math
import random
# import cairosvg
import numpy as np
from PIL import Image
from io import BytesIO
import pickle

from scipy import ndimage

from ts_loader import TSLoader
from image_augmentor import ImageAugmentor
from svgToPng import resize_and_center_image
from sklearn.preprocessing import StandardScaler


def get_number_images_per_class(input_folder):
    class_dic = {}

    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)

        # Check if the path is a directory
        if os.path.isdir(folder_path):
            # Iterate through SVG files in the current folder
            for svg_file in os.listdir(folder_path):
                if svg_file.endswith(".svg"):
                    ts_class = svg_file.split("-")[0]
                    if ts_class in class_dic:
                        class_dic[ts_class] += 1
                    else:
                        class_dic[ts_class] = 1
    return class_dic


def get_dataset_from_svg(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    rotation_range = [-15, 15]
    shear_range = [-0.2 * 180 / math.pi, 0.2 * 180 / math.pi]
    zoom_range = [0.8, 1.2]
    shift_range = [0.3, 0.3]

    augmentor = ImageAugmentor(0.5, shear_range, rotation_range, shift_range, zoom_range)
    # Define the size for the output PNG files
    output_size = (128, 128)
    class_dic = get_number_images_per_class(input_folder)

    # Iterate through the folders (1xx, 2xx, 3xx, 4xx, 5xx)
    # for folder_name in os.listdir(input_folder):
    #     folder_path = os.path.join(input_folder, folder_name)
    #
    #     # Check if the path is a directory
    #     if os.path.isdir(folder_path):
    #         # Create a subfolder in the output folder for each category
    #
    #         # Iterate through SVG files in the current folder
    #         for svg_file in os.listdir(folder_path):
    #             if svg_file.endswith(".svg"):
    #                 svg_path = os.path.join(folder_path, svg_file)
    #
    #                 ts_class = svg_file.split("-")[0]
    #                 output_category_folder = os.path.join(output_folder, ts_class)
    #                 os.makedirs(output_category_folder, exist_ok=True)
    #                 number_of_augmentations_from_image = 100 / class_dic[ts_class]
    #
    #                 with open(os.path.join(folder_path, svg_file), 'rb') as f:
    #                     svg_string = f.read()
    #                 png_bytes = cairosvg.surface.PNGSurface.convert(
    #                     bytestring=svg_string,
    #                     width=None,  # Use the SVG's original width
    #                     height=None  # Use the SVG's original height
    #                 )
    #                 print(ts_class)
    #                 for i in range(int(number_of_augmentations_from_image)):
    #                     background_index = random.randint(1, 18)
    #                     background_image = Image.open(f'backgrounds/background-{background_index}.png')
    #
    #                     svg_image = Image.open(BytesIO(png_bytes))
    #
    #                     # Resize and center the SVG image
    #                     first_aug, str_to_append = augmentor.get_random_transform_single_image_transformation(svg_image)
    #                     centered_svg = resize_and_center_image(first_aug)
    #                     background_image.paste(centered_svg, (0, 0), centered_svg)
    #                     np_image = np.asarray(background_image)
    #                     final_image, str_to_append = augmentor.get_random_transform_single_image_transformation_after_background(
    #                         np_image, str_to_append)
    #                     # Paste with transparency
    #                     # Define the output file path
    #                     output_file_path = os.path.join(output_category_folder,
    #                                                     f"{svg_file.strip('.svg')}-{str_to_append}.png")
    #                     pil_image = Image.fromarray(final_image)
    #                     pil_image.save(output_file_path)


def apply_sobel_filter(image, scaling=False):
    """Applies a Sobel filter to a PIL Image.

    Args:
        image: The PIL Image object to process.

    Returns:
        PIL Image: The filtered image.
    """

    # Convert to grayscale for Sobel filter
    image = image.convert('L')

    # Convert to a NumPy array for calculations
    image_array = np.array(image)

    image_array = image_array / image_array.std() - image_array.mean()

    # Define Sobel filter kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Calculate image gradients in x and y directions
    image_x = ndimage.convolve(image_array, sobel_x)
    image_y = ndimage.convolve(image_array, sobel_y)

    # Calculate gradient magnitude
    gradient_magnitude = np.hypot(image_x, image_y)
    if scaling:
        # Scale gradient magnitude to 0-255 range for image display
        gradient_magnitude = gradient_magnitude / gradient_magnitude.max() * 255

        # Create a new PIL Image from the results
        filtered_image = Image.fromarray(gradient_magnitude.astype(np.uint8))
        return filtered_image
    else:
        return gradient_magnitude


def get_dataset_normalized_sobel(input_folder, output_folder, images_rescaled=True):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    if images_rescaled:
        for file in os.listdir(input_folder):
            if file.endswith(".png"):
                image = Image.open(os.path.join(input_folder, file))
                edited_image = apply_sobel_filter(image)
                output_file_path = os.path.join(output_folder, file)
                edited_image.save(output_file_path)
    else:
        for folder_name in os.listdir(input_folder):
            folder_path = os.path.join(input_folder, folder_name)
            # Check if the path is a directory
            if os.path.isdir(folder_path):
                for folder2 in os.listdir(folder_path):
                    folder_path2 = os.path.join(folder_path, folder2)
                    if os.path.isdir(folder_path2):
                        for file in os.listdir(folder_path2):
                            if file.endswith(".png"):
                                output_category_folder = os.path.join(output_folder, folder_name, folder2)
                                os.makedirs(output_category_folder, exist_ok=True)
                                image = Image.open(os.path.join(folder_path2, file))
                                edited_image = apply_sobel_filter(image)
                                output_file_path = os.path.join(output_category_folder, file)
                                edited_image.save(output_file_path)


def rescale_image(image_array, old_min, old_max):
    new_min = 0
    new_max = 255
    rescaled_array = (((image_array - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
    return rescaled_array.astype(np.uint8)


def get_dataset_scalers(input_folder, input_rescaled_folder, grayscale=True):
    images = []
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        # Check if the path is a directory
        if os.path.isdir(folder_path):
            for folder2 in os.listdir(folder_path):
                folder_path2 = os.path.join(folder_path, folder2)
                if os.path.isdir(folder_path2):
                    for file in os.listdir(folder_path2):
                        if file.endswith(".png"):
                            images.append(os.path.join(folder_path2, file))

    images_rescaled = []
    for image in os.listdir(input_rescaled_folder):
        image_path = os.path.join(input_rescaled_folder, image)
        # Check if the path is a directory
        if image.endswith(".png"):
            images_rescaled.append(image_path)

    images_united = images + images_rescaled
    print("Image loading completed")

    if grayscale:
        images = [np.zeros((len(images_united),
                            128, 128)) for _ in range(1)][0]
        for i, image in enumerate(images_united):
            images[i] = apply_sobel_filter(Image.open(image), scaling=False)
        print("Applied sobel filter and slaying")

        images_reshaped = images.reshape(-1, 1)  # Reshape for a single channel
        print("Reshaped")
        scaler = StandardScaler()
        rescaled_images = scaler.fit(images_reshaped)
        with open('scaler_4.pkl', 'wb') as f:
            pickle.dump(rescaled_images, f)
    else:
        ts_loader = TSLoader('ts_4', False, 32)
        image_old_arrays = ts_loader.convert_path_list_to_images_and_labels_rgb_singlearray(images_united)
        for channel in range(3):
            images_reshaped = image_old_arrays[..., channel].reshape(-1, 1)  # Reshape for a single channel
            scaler = StandardScaler()
            rescaled_channel = scaler.fit(images_reshaped)  # Fit and transform
            with open(f'scaler_{channel}.pkl', 'wb') as f:
                pickle.dump(rescaled_channel, f)


def get_dataset_rgb_normalized(input_folder, input_rescaled_folder, output_folder, output_rescaled_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_rescaled_folder, exist_ok=True)
    # load paths of images from both directories
    images = []
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        # Check if the path is a directory
        if os.path.isdir(folder_path):
            for folder2 in os.listdir(folder_path):
                folder_path2 = os.path.join(folder_path, folder2)
                if os.path.isdir(folder_path2):
                    for file in os.listdir(folder_path2):
                        if file.endswith(".png"):
                            images.append(os.path.join(folder_path2, file))

    images_rescaled = []
    for image in os.listdir(input_rescaled_folder):
        image_path = os.path.join(input_rescaled_folder, image)
        # Check if the path is a directory
        if image.endswith(".png"):
            images_rescaled.append(image_path)

    images_united = images + images_rescaled

    ts_loader = TSLoader('ts_4', False, 32)
    image_old_arrays = ts_loader.convert_path_list_to_images_and_labels_rgb_singlearray(images_united)
    scaler = StandardScaler()
    image_arrays = [np.zeros((len(images_united),
                              128, 128, 3)) for _ in range(1)][0].astype(np.float64)
    for channel in range(3):
        images_reshaped = image_old_arrays[..., channel].reshape(-1, 1)  # Reshape for a single channel
        rescaled_channel = scaler.fit_transform(images_reshaped)  # Fit and transform
        image_arrays[..., channel] = rescaled_channel.reshape(65, 128, 128)  # Reshape back

    images_new = [np.zeros((len(images_united),
                            128, 128, 3)) for _ in range(1)][0].astype(np.uint8)
    for channel in range(3):
        channel_min = image_arrays[..., channel].min()
        channel_max = image_arrays[..., channel].max()
        images_new[..., channel] = rescale_image(image_arrays[..., channel], channel_min, channel_max)

    for i, image in enumerate(images_new):
        if i < len(images):
            input_path = images[i]
            output_file_path = input_path.replace(input_folder, output_folder)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            normalized_image = Image.fromarray(image)
            normalized_image.save(output_file_path)
        else:
            input_path = images_rescaled[i - len(images)]
            output_file_path = input_path.replace(input_rescaled_folder, output_rescaled_folder)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            normalized_image = Image.fromarray(image)
            normalized_image.save(output_file_path)


if __name__ == "__main__":
    input_folder = "cropped_images_test"
    input_rescaled_folder = "cropped_images_test"
    output_folder = "ts_4_test_normalized"
    output_rescaled_folder = "cropped_images_test_normalized"
    get_dataset_scalers(input_folder, input_rescaled_folder)
    # get_dataset_normalized_sobel(input_folder, output_folder, images_rescaled=True)
    # get_dataset_rgb_normalized(input_folder, input_rescaled_folder, output_folder, output_rescaled_folder)
    print("Conversion completed.")
