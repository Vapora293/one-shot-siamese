import os
import random
from PIL import Image
from svglib.svglib import svg2rlg
from svgToPng import resize_and_center_image
import math
from image_augmentor import ImageAugmentor
from io import BytesIO

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



input_folder = "official_traffic_sign_database//VL_6.1-2023_01-Vektorová_grafika"
output_folder = "output_png"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

rotation_range = [-15, 15]
shear_range = [-0.3 * 180 / math.pi, 0.3 * 180 / math.pi]
zoom_range = [0.8, 2]
shift_range = [5, 5]

augmentor = ImageAugmentor(0.5, shear_range, rotation_range, shift_range, zoom_range)
# Define the size for the output PNG files
output_size = (128, 128)
class_dic = get_number_images_per_class(input_folder)

background_images = []
for i in range(18):
    background_images.append(Image.open(f'backgrounds/background-{i}.png'))

# Iterate through the folders (1xx, 2xx, 3xx, 4xx, 5xx)
for folder_name in os.listdir(input_folder):
    folder_path = os.path.join(input_folder, folder_name)

    # Check if the path is a directory
    if os.path.isdir(folder_path):
        # Create a subfolder in the output folder for each category

        # Iterate through SVG files in the current folder
        for svg_file in os.listdir(folder_path):
            if svg_file.endswith(".svg"):
                svg_path = os.path.join(folder_path, svg_file)

                ts_class = svg_file.split("-")[0]
                output_category_folder = os.path.join(output_folder, ts_class)
                os.makedirs(output_category_folder, exist_ok=True)
                number_of_augmentations_from_image = 100 / class_dic[ts_class]

                with open(svg_file, 'rb') as f:
                    svg_string = f.read()
                png_bytes = cairosvg.surface.PNGSurface.convert(
                    bytestring=svg_string,
                    width=None,  # Use the SVG's original width
                    height=None  # Use the SVG's original height
                )
                for i in range(int(number_of_augmentations_from_image)):
                    print(ts_class)
                    background_index = random.randint(1, 18)
                    background_file = f'backgrounds/background-{background_index}.png'
                    background_image = background_images[background_index]

                    svg_image = Image.open(io.BytesIO(png_bytes))

                    # Resize and center the SVG image
                    centered_svg = resize_and_center_image(svg_image)
                    background_image.paste(centered_svg, (0, 0), centered_svg)  # Paste with transparency
                    drawing = svg2rlg(svg_path)
                    drawing_width = drawing.width
                    drawing_height = drawing.height
                    final_image, str_to_append = augmentor.get_random_transform_single_image_transformation(background_image)
                    # Define the output file path
                    output_file_path = os.path.join(output_category_folder,
                                                f"{ts_class}-{svg_file.strip('.svg')}-{str_to_append}.png")


print("Conversion completed.")