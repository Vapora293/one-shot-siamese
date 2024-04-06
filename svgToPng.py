from PIL import Image, ImageFilter
from scipy import ndimage
import numpy as np
import os
import random
import io
from sklearn.preprocessing import RobustScaler


def resize_and_center_image(image, max_size=128):
    """Resizes an image to fit within a maximum dimension while maintaining aspect ratio, then centers it on a square canvas.

    Args:
        image (Image.Image): The PIL image to resize and center.
        max_size (int, optional): The maximum width or height of the resized image. Defaults to 128.

    Returns:
        Image.Image: The resized and centered image.
    """

    width, height = image.size

    # Resize while maintaining aspect ratio
    if width > height:
        scale_factor = max_size / width
    else:
        scale_factor = max_size / height

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized_image = image.resize((new_width, new_height))

    # Center on a square canvas
    result_image = Image.new('RGBA', (max_size, max_size), (0, 0, 0, 0))  # Transparent background
    x_offset = (max_size - new_width) // 2
    y_offset = (max_size - new_height) // 2
    result_image.paste(resized_image, (x_offset, y_offset))

    return result_image


def iterateAmongAllPngsResize():
    # iterate through all the directories in the input folder. For every png you find, convert it to jpg using the code below
    input_folder = "ts_new_copy"
    output_folder = "ts_new_copy"

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for folder in os.listdir(input_folder):
        for folder_second in os.listdir(os.path.join(input_folder, folder)):
            for filename in os.listdir(os.path.join(input_folder, folder, folder_second)):
                if filename.endswith(".png"):
                    background_indexes = [random.randint(1, 18) for _ in range(10)]
                    for index in background_indexes:
                        background = Image.open(os.path.join('backgrounds', f"background-{index}.png"))
                        # img = Image.open(os.path.join(input_folder, folder, folder_second, filename))
                        img = Image.open("test.png")
                        width = img.size[0]
                        height = img.size[1]
                        img.resize()
                    bigside = width if width > height else height
                    # background = Image.new('RGBA', (bigside, bigside), (255, 255, 255))
                    offset = (int(round(((bigside - width) / 2), 0)), int(round(((bigside - height) / 2), 0)))
                    background.paste(img, offset)
                    img = background
                    img = img.resize((128, 128))
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
                    os.rename(os.path.join(input_folder, folder, folder_second, filename),
                              os.path.join(input_folder, folder, folder_second, filename.replace(",", ".")))
    print("Conversion completed.")


def getPngBytesFromSVG():
    svg_file = "official_traffic_sign_database/VL_6.1-2023_01-Vektorová_grafika/4xx/424-22.svg"
    # Load SVG as a surface
    with open(svg_file, 'rb') as f:
        svg_string = f.read()
    png_bytes = cairosvg.surface.PNGSurface.convert(
        bytestring=svg_string,
        width=None,  # Use the SVG's original width
        height=None  # Use the SVG's original height
    )

    # svg_image = cv2.imread(png_bytes, cv2.IMREAD_UNCHANGED)

    # background_file = 'backgrounds/background-1.png'
    # Load background image
    # background_image = Image.open(background_file)

    svg_image = Image.open(io.BytesIO(png_bytes))

    centered_svg = resize_and_center_image(svg_image)
    centered_svg.save("result_trnsp.png")


def testGrayImages(dirToLoad):
    scaler = RobustScaler()
    for filename in os.listdir(dirToLoad):
        image = Image.open(os.path.join(dirToLoad, filename)).convert('L')  # Convert to grayscale ('L' mode)
        image = np.asarray(image).astype(np.float64)
        # image = image / image.std() - image.mean()
        image = (image - image.min()) / (image.max() - image.min())  # Scale to 0-1
        # image = scaler.fit_transform(image)
        image = (image * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8

        grayscale_pil_image = Image.fromarray(image.astype(np.float64)).convert('L')
        grayscale_pil_image.save(os.path.abspath(str(filename) + 'test.png'))


def apply_sobel_filter(image):
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

    # Scale gradient magnitude to 0-255 range for image display
    gradient_magnitude = gradient_magnitude / gradient_magnitude.max() * 255

    # Create a new PIL Image from the results
    filtered_image = Image.fromarray(gradient_magnitude.astype(np.uint8))

    return filtered_image


def oldFunctions():
    pass
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


if __name__ == "__main__":
    # dirToLoad = ["ts_4\\validation\\120", "ts_4\\validation\\273", "ts_4\\validation\\320", "ts_4\\validation\\424"]
    dirToLoad = ["C:\\Users\\filip\\FIIT\\Bakalarka\\frame_sample_wrapped\\cropped_images"]
    global_index = 0
    for directory in dirToLoad:
        for index, image_path in enumerate(os.listdir(directory)):
            if index > 20:
                global_index += index
                break
            original_image = Image.open(os.path.join(directory, image_path))

            filtered_image = apply_sobel_filter(original_image)
            filtered_image.save(f'cropped_images_sobel_normalize_{global_index + index}.png')
