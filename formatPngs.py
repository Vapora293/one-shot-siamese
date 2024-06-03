import os
from PIL import Image
import os
import cairosvg
import cv2
import numpy as np
import random
import io

input_folder = "ts_5"
evaluation_dataset_path = "ts_5/validation"
output_folder = "ts_5_output"


def iterateThroughAllSubfoldersAndExtractAllPngsToOutputFolder():
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        # if folder
        if os.path.isdir(file_path):
            # iterate around all png files in the folder
            for png in os.listdir(file_path):
                if png.endswith(".png"):
                    # if the dir named as a second element after splitting its name based on a - doesn't exist create it
                    ts_type = "unknown"
                    if png.__contains__("-"):
                        ts_type = png.split("-")[0].strip(".png")
                    else:
                        ts_type = png.strip(".png")
                    future_path = os.path.join(output_folder, ts_type)
                    if not os.path.exists(future_path):
                        os.makedirs(future_path)
                    former_path = os.path.join(input_folder, filename, png)
                    new_path = os.path.join(output_folder, ts_type, png)
                    os.rename(former_path, new_path)


def convertToPng():
    for png in os.listdir(input_folder):
        # file_path = os.path.join(folder, filename)
        # if folder
        # if os.path.isdir(file_path):
        # iterate around all png files in the folder
        # for png in os.listdir(file_path):
        if png.endswith(".bmp"):
            # if the dir named as a second element after splitting its name based on a - doesn't exist create it
            # ts_type = png.split("-")[1].strip(".png")
            ts_type = png.split("_")[0].strip(".bmp")
            future_path = os.path.join(input_folder, ts_type)
            if not os.path.exists(future_path):
                os.makedirs(future_path)
            # former_path = os.path.join(input_folder, filename, png)
            former_path = os.path.join(input_folder, png)
            png = png.strip(png.split("_")[0])[1:]
            new_path = os.path.join(input_folder, ts_type, png)
            os.rename(former_path, new_path)


def getEvalDataset():
    eval_dataset = []
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isdir(file_path):
            appendedImages = 0
            desiredNumber = 0
            if (len(os.listdir(file_path)) > 2):
                desiredNumber = ((len(os.listdir(file_path)) / 10) * 2)
                if desiredNumber < 1 or len(os.listdir(file_path)) == 3:
                    desiredNumber = 1
            for png in os.listdir(file_path):
                if appendedImages <= desiredNumber and desiredNumber != 0:
                    eval_dataset.append(os.path.join(file_path, png))
                    appendedImages += 1

    for image in eval_dataset:
        image_stripped_path = image.strip(input_folder)
        # join eval path and the stripped path
        imageFullPath = os.path.join(evaluation_dataset_path, *image_stripped_path.split(os.path.sep))
        # If the folder doesn't exist create it
        if not os.path.exists(os.path.dirname(imageFullPath)):
            os.makedirs(os.path.dirname(imageFullPath))
        os.rename(image, imageFullPath)


def resize_and_center_image(image, target_size=128, fill_color=(0, 0, 0, 0)):  # Default transparent
    """Resizes an image, either downscaling to fit within a target dimension or upscaling
    to reach the target dimension while maintaining aspect ratio. Then centers it on a
    square canvas with a specified fill color.

    Args:
        image (Image.Image): The PIL image to resize and center.
        target_size (int, optional): The target maximum width or height. Defaults to 128.
        fill_color (tuple, optional): RGBA tuple representing the fill color.
                                      Defaults to (0, 0, 0, 0) for transparent.

    Returns:
        Image.Image: The resized and centered image.
    """

    width, height = image.size

    # Determine scaling mode (upscale or downscale)
    if max(width, height) < target_size:
        mode = 'upscale'
    else:
        mode = 'downscale'

    # Calculate scaling factor
    if width > height:
        scale_factor = target_size / width
    else:
        scale_factor = target_size / height

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Perform resizing based on mode
    if mode == 'upscale':
        resized_image = image.resize((new_width, new_height), resample=Image.LANCZOS)  # High-quality upscaling
    else:  # 'downscale'
        resized_image = image.resize((new_width, new_height))

    # Center on a square canvas
    result_image = Image.new('RGBA', (target_size, target_size), fill_color)
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    result_image.paste(resized_image, (x_offset, y_offset))

    return result_image


def iterateAmongAllPngsResize():
    # iterate through all the directories in the input folder. For every png you find, convert it to jpg using the code below

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for folder in os.listdir(input_folder):
        for folder_second in os.listdir(os.path.join(input_folder, folder)):
            print(f"Starting resizing class {folder_second}")
            for filename in os.listdir(os.path.join(input_folder, folder, folder_second)):
                if filename.endswith(".jpg"):
                    # background_indexes = [random.randint(1, 18) for _ in range(10)]
                    # for index in background_indexes:
                    #     # background = Image.open(os.path.join('backgrounds', f"background-{index}.png"))
                    img = Image.open(os.path.join(input_folder, folder, folder_second, filename))
                    # img = Image.open("test.png")
                    img = resize_and_center_image(img)
                    image_output_folder = os.path.join(output_folder, folder, folder_second)
                    os.makedirs(image_output_folder, exist_ok=True)
                    img.save(os.path.join(image_output_folder, filename))
    print("Conversion completed.")


def replaceAllCommasWithStopsInAllFileNamesInFolder():
    # iterate through all the directories in the input folder. For every png you find, convert it to jpg using the code below

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for folder in os.listdir(input_folder):
        for folder_second in os.listdir(os.path.join(input_folder, folder)):
            for filename in os.listdir(os.path.join(input_folder, folder, folder_second)):
                if "," in filename:
                    os.rename(os.path.join(input_folder, folder, folder_second, filename),
                              os.path.join(input_folder, folder, folder_second, filename.replace(",", ".")))
    print("Conversion completed.")


if __name__ == "__main__":
    # convertToPng()
    # iterateAmongAllPngsResize()
    getEvalDataset()

    # svg_file = "official_traffic_sign_database/VL_6.1-2023_01-Vektorová_grafika/4xx/424-22.svg"
# # Load SVG as a surface
# with open(svg_file, 'rb') as f:
#     svg_string = f.read()
# png_bytes = cairosvg.surface.PNGSurface.convert(
#     bytestring=svg_string,
#     width=None,  # Use the SVG's original width
#     height=None  # Use the SVG's original height
# )
#
# # svg_image = cv2.imread(png_bytes, cv2.IMREAD_UNCHANGED)
#
# # background_file = 'backgrounds/background-1.png'
# # Load background image
# # background_image = Image.open(background_file)
#
# svg_image = Image.open(io.BytesIO(png_bytes))
#
# centered_svg = resize_and_center_image(svg_image)
# centered_svg.save("result_trnsp.png")

# Resize and center the SVG image

# background_image.paste(centered_svg, (0, 0), centered_svg)  # Paste with transparency
# Paste onto the background

# background_image.save("result_trnsp.png")
# Save the result

# # Extract relevant dimensions
# svg_height, svg_width, _ = png_array.shape
#
# # Region of Interest (ROI) in the background image
# roi = background_image[y_offset:y_offset + svg_height, x_offset:x_offset + svg_width]
#
# # Alpha blending for transparency
# alpha = png_array[:, :, 3] / 255.0
# for c in range(0, 3):
#     roi[alpha > 0, c] = alpha[alpha > 0] * png_array[alpha > 0, c] + (1 - alpha[alpha > 0]) * roi[alpha > 0, c]
#
# # Store modified background (if needed)
# cv2.imwrite("result_image.png", background_image)

# Composite SVG onto the background (assuming SVG has transparency)
# alpha = png_array[:, :, 3] / 255.0
# for ch in range(3):
#     background_image[alpha > 0, ch] = (alpha[alpha > 0] * svg_array[alpha > 0, ch] +
#                                       (1 - alpha[alpha > 0]) * background_image[alpha > 0, ch])
#
# # Save the result as PNG
# cv2.imwrite("dog_on_background.png", background_image)

# import os
# from PIL import Image
# from svglib.svglib import svg2rlg
# from io import BytesIO
#
# input_folder = "official_traffic_sign_database//VL_6.1-2023_01-Vektorová_grafika"
# output_folder = "output_png"
#
# # Create the output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)
#
# # Define the size for the output PNG files
# output_size = (128, 128)
#
# # Iterate through the folders (1xx, 2xx, 3xx, 4xx, 5xx)
# for folder_name in os.listdir(input_folder):
#     folder_path = os.path.join(input_folder, folder_name)
#
#     # Check if the path is a directory
#     if os.path.isdir(folder_path):
#         # Create a subfolder in the output folder for each category
#         output_category_folder = os.path.join(output_folder, folder_name)
#         os.makedirs(output_category_folder, exist_ok=True)
#
#         # Iterate through SVG files in the current folder
#         for svg_file in os.listdir(folder_path):
#             if svg_file.endswith(".svg"):
#                 svg_path = os.path.join(folder_path, svg_file)
#
#                 # Define the output file path
#                 output_file_path = os.path.join(output_category_folder, f"{folder_name}-{svg_file.replace('.svg', '.png')}")
#
#                 # Convert SVG to PNG using svglib and Pillow
#                 drawing = svg2rlg(svg_path)
#                 drawing_width = drawing.width
#                 drawing_height = drawing.height
#
#                 # Create a blank image with the desired size
#                 img = Image.new("RGB", output_size, "white")
#
#                 # Scale and paste the SVG onto the blank image
#                 scale_factor = min(output_size[0] / drawing_width, output_size[1] / drawing_height)
#                 new_width = int(drawing_width * scale_factor)
#                 new_height = int(drawing_height * scale_factor)
#
#                 drawing_scale = drawing.scale(new_width / drawing_width, new_height / drawing_height)
#                 drawing_scale.width = new_width
#                 drawing_scale.height = new_height
#
#                 drawing_scale.drawOn(img, 0, 0)
#
#                 # Save the resulting image
#                 img.save(output_file_path)
#
# print("Conversion completed.")


# import os
# import aspose.words as aw
#
# input_folder = "official_traffic_sign_database//VL_6.1-2023_01-Vektorová_grafika"
# output_folder = "official_ts_jpg"
#
# # Create the output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)
#
# # Define the size for the output PNG files
# output_size = (128, 128)
#
# # Iterate through the folders (1xx, 2xx, 3xx, 4xx, 5xx)
# for folder_name in os.listdir(input_folder):
#     folder_path = os.path.join(input_folder, folder_name)
#
#     # Check if the path is a directory
#     if os.path.isdir(folder_path):
#         # Create a subfolder in the output folder for each category
#         output_category_folder = os.path.join(output_folder, folder_name)
#         os.makedirs(output_category_folder, exist_ok=True)
#
#         # Iterate through SVG files in the current folder
#         for svg_file in os.listdir(folder_path):
#             if svg_file.endswith(".svg"):
#                 svg_path = os.path.join(folder_path, svg_file)
#
#                 # Define the output file path
#                 output_file_path = os.path.join(output_category_folder, f"{folder_name}-{svg_file.replace('.svg', '.jpg')}")
#                 doc = aw.Document()
#
#                 # create a document builder and initialize it with document object
#                 builder = aw.DocumentBuilder(doc)
#
#                 # insert SVG image to document
#                 shape = builder.insert_image(svg_path)
#
#                 # OPTIONAL
#                 # Calculate the maximum width and height and update page settings
#                 # to crop the document to fit the size of the pictures.
#                 pageSetup = builder.page_setup
#                 pageSetup.page_width = shape.width
#                 pageSetup.page_height = shape.height
#                 pageSetup.top_margin = 0
#                 pageSetup.left_margin = 0
#                 pageSetup.bottom_margin = 0
#                 pageSetup.right_margin = 0
#
#                 doc.save(output_file_path)
#
# print("Conversion completed.")

# from PIL import Image

# image = Image.open("C:\\Users\\filip\\PycharmProjects\\Siamese-Networks-for-One-Shot-Learning\\official_traffic_sign_database\\VL_6.1-2023_01-Vektorová_grafika\\1xx\\112-10.png")
# print("alles clar")
# # import os
# # os.environ['path'] += r';C:\Users\filip\PycharmProjects\Siamese-Networks-for-One-Shot-Learning\libcairo-2.dll'
# # import cairosvg
# #
# # input_folder = "official_traffic_sign_database//VL_6.1-2023_01-Vektorová_grafika"
# # output_folder = "official_ts_cairo"
# #
# # # Create the output folder if it doesn't exist
# # os.makedirs(output_folder, exist_ok=True)
# #
# # # Iterate through the folders (1xx, 2xx, 3xx, 4xx, 5xx)
# # for folder_name in os.listdir(input_folder):
# #     folder_path = os.path.join(input_folder, folder_name)
#
#     # Check if the path is a directory
#     if os.path.isdir(folder_path):
#         # Create a subfolder in the output folder for each category
#         output_category_folder = os.path.join(output_folder, folder_name)
#         os.makedirs(output_category_folder, exist_ok=True)
#
#         # Iterate through SVG files in the current folder
#         for svg_file in os.listdir(folder_path):
#             if svg_file.endswith(".svg"):
#                 svg_path = os.path.join(folder_path, svg_file)
#
#                 # Define the output file path
#                 output_file_path = os.path.join(output_category_folder, f"{folder_name}-{svg_file.replace('.svg', '.png')}")
#
#                 # Convert SVG to PNG using cairosvg
#                 cairosvg.svg2png(url=svg_path, write_to=output_file_path, output_width=128, output_height=128)
#
# print("Conversion completed.")

# import aspose.words as aw

# iterateAmongAllPngsResize()