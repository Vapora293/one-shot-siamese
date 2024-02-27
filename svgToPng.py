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
import os
from PIL import Image


def iterateAmongAllPngsResize():
    # iterate through all the directories in the input folder. For every png you find, convert it to jpg using the code below
    input_folder = "ts_new"
    output_folder = "ts_new"

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for folder in os.listdir(input_folder):
        for folder_second in os.listdir(os.path.join(input_folder, folder)):
            for filename in os.listdir(os.path.join(input_folder, folder, folder_second)):
                if filename.endswith(".png"):
                    img = Image.open(os.path.join(input_folder, folder, folder_second, filename))
                    width = img.size[0]
                    height = img.size[1]
                    if (width != height):
                        bigside = width if width > height else height

                        background = Image.new('RGBA', (bigside, bigside), (255, 255, 255, 255))
                        offset = (int(round(((bigside - width) / 2), 0)), int(round(((bigside - height) / 2), 0)))
                        background.paste(img, offset)
                        img = background
                    img = img.resize((256, 256))
                    img.save(os.path.join(output_folder, folder, folder_second, filename))
    print("Conversion completed.")

def replaceAllCommasWithStopsInAllFileNamesInFolder():
    # iterate through all the directories in the input folder. For every png you find, convert it to jpg using the code below
    input_folder = "ts_new"
    output_folder = "ts_new"

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for folder in os.listdir(input_folder):
        for folder_second in os.listdir(os.path.join(input_folder, folder)):
            for filename in os.listdir(os.path.join(input_folder, folder, folder_second)):
                if "," in filename:
                    os.rename(os.path.join(input_folder, folder, folder_second, filename), os.path.join(input_folder, folder, folder_second, filename.replace(",", ".")))
    print("Conversion completed.")

replaceAllCommasWithStopsInAllFileNamesInFolder()