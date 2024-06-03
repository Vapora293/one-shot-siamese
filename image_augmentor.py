import numpy as np
import scipy.ndimage as ndi
import math
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageEnhance


class ImageAugmentor:
    """Class that performs image augmentation.

    Big part of this code uses Keras ImageDataGenerator file code

    Attributes:
        augmentation_probability: probability of augmentation
        shear_range: shear intensity (shear angle in degrees).
        rotation_range: degrees (0 to 180).
        shift_range: fraction of total shift (horizontal and vertical).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
    """

    def __init__(self, augmentation_probability, shear_range, rotation_range, shift_range, zoom_range):
        """Inits ImageAugmentor with the provided values for the attributes."""
        self.augmentation_probability = augmentation_probability
        self.shear_range = shear_range
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range

    def _transform_matrix_offset_center(self, transformation_matrix, width, height):
        """ Corrects the offset of tranformation matrix

            Corrects the offset of tranformation matrix for the specified image
            dimensions by considering the center of the image as the central point

            Args:
                transformation_matrix: transformation matrix from a specific
                    augmentation.
                width: image width
                height: image height

            Returns:
                The corrected transformation matrix.
        """

        o_x = float(width) / 2 + 0.5
        o_y = float(height) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transformation_matrix = np.dot(
            np.dot(offset_matrix, transformation_matrix), reset_matrix)

        return transformation_matrix

    # Applies a provided transformation to the image
    def _apply_transform(self, image, transformation_matrix):
        """ Applies a provided transformation to the image

            Args:
                image: image to be augmented
                transformation_matrix: transformation matrix from a specific
                    augmentation.

            Returns:
                The transformed image
        """

        channel_axis = 2
        image = np.rollaxis(image, channel_axis, 0)
        final_affine_matrix = transformation_matrix[:2, :2]
        final_offset = transformation_matrix[:2, 2]

        channel_images = [ndi.interpolation.affine_transform(
            image_channel,
            final_affine_matrix,
            final_offset,
            order=0,
            mode='nearest',
            cval=0) for image_channel in image]

        image = np.stack(channel_images, axis=0)
        image = np.rollaxis(image, 0, channel_axis + 1)

        return image

    def _perform_random_rotation(self, image):
        """ Applies a random rotation

            Args:
                image: image to be augmented

            Returns:
                The transformed image
        """

        theta = np.deg2rad(np.random.uniform(
            low=self.rotation_range[0], high=self.rotation_range[1]))

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])

        transformation_matrix = self._transform_matrix_offset_center(
            rotation_matrix, image.shape[0], image.shape[1])
        image = self._apply_transform(image, transformation_matrix)

        return image

    def _perform_random_rotation_cv(self, image):
        """Applies a random rotation to an OpenCV image, filling missing pixels with transparency.

        Args:
            image: The OpenCV image with an alpha channel (4 channels).

        Returns:
            The rotated OpenCV image with transparency preserved.
        """

        height, width = image.shape[:2]
        theta = np.random.uniform(low=self.rotation_range[0], high=self.rotation_range[1])
        theta_rad = np.deg2rad(theta)

        # Calculate necessary dimensions for expanded canvas
        abs_cos_theta = abs(np.cos(theta_rad))
        abs_sin_theta = abs(np.sin(theta_rad))
        new_width = int(width * abs_cos_theta + height * abs_sin_theta)
        new_height = int(width * abs_sin_theta + height * abs_cos_theta)

        # Create transparent background
        background = np.zeros((new_height, new_width, 4), dtype=np.uint8)  # 4 channels

        # Center original image on the transparent background
        x_offset = (new_width - width) // 2
        y_offset = (new_height - height) // 2
        background[y_offset:y_offset + height, x_offset:x_offset + width] = image

        # Calculate rotation matrix and apply rotation
        rotation_matrix = cv2.getRotationMatrix2D((new_width / 2, new_height / 2), theta, 1.0)
        rotated_image = cv2.warpAffine(background, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_TRANSPARENT)

        return rotated_image

    def _perform_random_rotation_pil(self, image):

        angle = np.random.uniform(low=self.rotation_range[0], high=self.rotation_range[1])

        # Calculate new dimensions, create transparent background (calculate precisely if needed)
        new_width = image.width
        new_height = image.height
        background = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))

        # Paste original image at the center and rotate
        x_offset = (new_width - image.width) // 2
        y_offset = (new_height - image.height) // 2
        background.paste(image, (x_offset, y_offset))
        rotated_image = background.rotate(angle, Image.NEAREST, expand=1)

        return rotated_image  # Convert back if needed

    def _perform_random_shear_pil(self, image):
        """Applies a random shear to a PIL Image, preserving transparency.

        Args:
            image: The PIL Image (RGBA mode).

        Returns:
            The sheared PIL Image.
        """

        width, height = image.size

        # Calculate shear and create transparent background
        shear_ang = math.radians(np.random.uniform(low=self.shear_range[0], high=self.shear_range[1]))
        abs_cos_shear = abs(np.cos(shear_ang))
        abs_sin_shear = abs(np.sin(shear_ang))
        new_width = int(width * abs_cos_shear + height * abs_sin_shear)
        new_height = int(width * abs_sin_shear + height * abs_cos_shear)
        background = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))

        # Center original image on transparent background
        x_offset = (new_width - width) // 2
        y_offset = (new_height - height) // 2
        background.paste(image, (x_offset, y_offset))

        # Calculate affine transformation for the shear
        transform_data = (1, math.tan(shear_ang), -x_offset * math.tan(shear_ang),
                          0, 1, 0)

        # Apply shear
        sheared_image = background.transform(background.size, Image.AFFINE, data=transform_data,
                                             resample=Image.NEAREST)

        return sheared_image

    def _perform_random_shift_pil(self, image):
        """Applies a random shift to a PIL Image, preserving transparency.

        Args:
            image: The PIL Image (RGBA mode).

        Returns:
            The shifted PIL Image.
        """

        width, height = image.size

        # Calculate random shifts
        tx = np.random.uniform(-self.shift_range[0] * width, self.shift_range[0] * width)
        ty = np.random.uniform(-self.shift_range[1] * height, self.shift_range[1] * height)

        # Create transparent background (slightly larger to accommodate shifts)
        expanded_width = width + int(abs(tx))
        expanded_height = height + int(abs(ty))
        background = Image.new('RGBA', (expanded_width, expanded_height), (0, 0, 0, 0))

        # Paste the original image with offset
        x_offset = int(max(tx, 0))
        y_offset = int(max(ty, 0))
        background.paste(image, (x_offset, y_offset))

        # Calculate affine transform for shift
        transform_data = (1, 0, tx, 0, 1, ty)

        # Apply shift
        shifted_image = background.transform(background.size, Image.AFFINE, data=transform_data,
                                             resample=Image.NEAREST)

        return shifted_image

    def _perform_random_zoom_pil(self, image):
        """Applies a random zoom to a PIL Image, preserving transparency.

        Args:
            image: The PIL Image (RGBA mode).

        Returns:
            The zoomed PIL Image.
        """

        width, height = image.size

        # Calculate zoom factors
        zx = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
        zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1])

        # Create transparent background (larger to accommodate zoom-out)
        new_width = int(width * max(1 / zx, 1))
        new_height = int(height * max(1 / zy, 1))
        background = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))

        # Center original image on the transparent background
        x_offset = (new_width - width) // 2
        y_offset = (new_height - height) // 2
        background.paste(image, (x_offset, y_offset))

        # Calculate affine transform for the zoom
        transform_data = (zx, 0, x_offset, 0, zy, y_offset)

        # Apply zoom
        zoomed_image = background.transform(background.size, Image.AFFINE, data=transform_data,
                                            resample=Image.NEAREST)

        return zoomed_image

    def _perform_random_shear_cv(self, image):
        """Applies a random shear to an OpenCV image, preserving transparency.

        Args:
            image: The OpenCV image with an alpha channel (4 channels).

        Returns:
            The sheared OpenCV image with transparency preserved.
        """

        height, width = image.shape[:2]

        # Calculate shear and create transparent background
        shear_ang = np.deg2rad(np.random.uniform(low=self.shear_range[0], high=self.shear_range[1]))
        abs_cos_shear = abs(np.cos(shear_ang))
        abs_sin_shear = abs(np.sin(shear_ang))
        new_width = int(width * abs_cos_shear + height * abs_sin_shear)
        new_height = int(width * abs_sin_shear + height * abs_cos_shear)
        background = np.zeros((new_height, new_width, 4), dtype=np.uint8)

        # Center original image on transparent background
        x_offset = (new_width - width) // 2
        y_offset = (new_height - height) // 2
        background[y_offset:y_offset + height, x_offset:x_offset + width] = image

        # Create shear matrix
        shear_matrix = np.array([
            [1, -np.sin(shear_ang), 0],
            [0, np.cos(shear_ang), 0],
            [0, 0, 1]
        ])

        # Apply shear using affine transformation
        transform_matrix = cv2.getAffineTransform(
            src=np.float32([[0, 0], [width, 0], [0, height]]),
            dst=np.float32([[0, 0], [width + shear_ang * height, 0], [shear_ang * height, height]])
        )

        sheared_image = cv2.warpAffine(
            background,
            transform_matrix,
            (new_width, new_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT
        )

        return sheared_image

    def _perform_random_shear(self, image):
        """ Applies a random shear

            Args:
                image: image to be augmented

            Returns:
                The transformed image
        """

        shear = np.deg2rad(np.random.uniform(
            low=self.shear_range[0], high=self.shear_range[1]))

        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        transformation_matrix = self._transform_matrix_offset_center(
            shear_matrix, image.shape[0], image.shape[1])
        image = self._apply_transform(image, transformation_matrix)

        return image

    def _perform_random_shift(self, image):
        """ Applies a random shift in x and y

            Args:
                image: image to be augmented

            Returns:
                The transformed image
        """

        tx = np.random.uniform(-self.shift_range[0],
                               self.shift_range[0])
        ty = np.random.uniform(-self.shift_range[1],
                               self.shift_range[1])

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        transformation_matrix = translation_matrix  # no need to do offset
        image = self._apply_transform(image, transformation_matrix)

        return image

    def _perform_random_zoom(self, image):
        """ Applies a random zoom

            Args:
                image: image to be augmented

            Returns:
                The transformed image
        """
        zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transformatiom_matrix = self._transform_matrix_offset_center(
            zoom_matrix, image.shape[0], image.shape[1])
        image = self._apply_transform(image, transformatiom_matrix)

        return image

    def _adjust_brightness_contrast(self, image, negative=False, contrast=False):
        """Adjusts the brightness and contrast of an image.

        Args:
            image: Input image as a NumPy array.
            brightness_factor: Factor to adjust brightness.
            contrast_factor: Factor to adjust contrast.

        Returns:
            NumPy array representing the image with adjusted brightness and contrast.
        """
        if (contrast):
            brightness_factor = np.random.random() * 0.4 + 0.8
            contrast_factor = np.random.random() * 0.5 + 0.3
        elif (negative):
            brightness_factor = np.random.random() * 0.5 + 0.3
            contrast_factor = np.random.random() * 0.3 + 0.7
        else:
            brightness_factor = np.random.random() * 0.4 + 1.3
            contrast_factor = np.random.random() * 0.3 + 0.7

        # Convert NumPy array to PIL Image
        pil_image = Image.fromarray(image)

        # Apply brightness and contrast adjustments
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness_factor)
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast_factor)

        # Convert PIL Image back to NumPy array
        return np.array(pil_image)

    def _add_fog_effect(self, image, intensity=0.9, white_threshold=0.98):
        """Simulates a fog effect on an image.

        Args:
            image: The input image as a NumPy array.
            intensity (float, optional): The intensity of the fog effect. Defaults to 0.5.

        Returns:
            A NumPy array representing the image with the fog effect applied.
        """

        # Convert image to HSV color space (better for color manipulation)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Extract channels
        hue_channel = hsv_image[:, :, 0]
        saturation_channel = hsv_image[:, :, 1]
        value_channel = hsv_image[:, :, 2]

        # Identify white pixel mask (all channels close to maximum)
        white_mask = (hue_channel >= (white_threshold) * 255) & \
                     (saturation_channel >= (white_threshold) * 255) & \
                     (value_channel >= (white_threshold) * 255)
        white_mask = white_mask.astype(np.uint8)  # Convert to uint8
        # Apply Gaussian blur (masked for non-white areas)
        blurred_value_channel = cv2.GaussianBlur(value_channel * (1 - white_mask), (7, 7), 0)

        # Increase intensity for non-white areas
        fog_mask = (1.0 + intensity) * blurred_value_channel

        # Merge channels back, preserving white pixels
        hsv_image[:, :, 0] = hue_channel
        hsv_image[:, :, 1] = saturation_channel
        hsv_image[:, :, 2] = value_channel * white_mask + fog_mask * (1 - white_mask)
        foggy_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        return foggy_image  # Return NumPy array in BGR format

    def _gaussian_blur(self, image):
        """Applies a Gaussian Blur to an image.

        Args:
            image: The input image as a NumPy array.
            kernel_size (tuple, optional): The size of the blurring kernel. Defaults to (5, 5).
            sigmaX (float, optional): Standard deviation in X direction. Defaults to 0.
            sigmaY (float, optional): Standard deviation in Y direction. Defaults to 0.

        Returns:
            A NumPy array representing the blurred image.
        """
        kernel_number = 5
        kernel_size = (kernel_number, kernel_number)
        sigmaX = 0
        sigmaY = 0
        return cv2.GaussianBlur(image, kernel_size, sigmaX, sigmaY)

    def _add_white_overlay(self, image, opacity=0.9):
        """Adds a white transparent layer on top of an image.

        Args:
            image: The input image as a NumPy array.
            opacity (float, optional): The transparency of the white layer (0.0 to 1.0). Defaults to 0.5.

        Returns:
            A NumPy array representing the image with the white overlay.
        """

        # Get image height and width
        height, width, channels = image.shape

        # Create a white mask with desired opacity
        white_mask = np.ones((height, width, 3), dtype=np.uint8) * 255  # White color
        white_mask[..., -1] = opacity * 255  # Set alpha channel for transparency

        # Add weighted sum with the image
        return cv2.addWeighted(image, 1.0, white_mask, 1.0, 0)

    def get_random_transform(self, images):
        """ Applies a random augmentation to pairs of images

            Args:
                images: pairs of the batch to be augmented

            Returns:
                The transformed images
        """

        number_of_pairs_of_images = images[0].shape[0]
        random_numbers = np.random.random(
            size=(number_of_pairs_of_images * 2, 4))

        for pair_index in range(number_of_pairs_of_images):
            image_1 = images[0][pair_index, :, :, :]
            image_2 = images[1][pair_index, :, :, :]

            if random_numbers[pair_index * 2, 0] > 0.5:
                image_1 = self._perform_random_rotation(image_1)
            if random_numbers[pair_index * 2, 1] > 0.5:
                image_1 = self._perform_random_shear(image_1)
            if random_numbers[pair_index * 2, 2] > 0.5:
                image_1 = self._perform_random_shift(image_1)
            if random_numbers[pair_index * 2, 3] > 0.5:
                image_1 = self._perform_random_zoom(image_1)

            if random_numbers[pair_index * 2 + 1, 0] > 0.5:
                image_2 = self._perform_random_rotation(image_2)
            if random_numbers[pair_index * 2 + 1, 1] > 0.5:
                image_2 = self._perform_random_shear(image_2)
            if random_numbers[pair_index * 2 + 1, 2] > 0.5:
                image_2 = self._perform_random_shift(image_2)
            if random_numbers[pair_index * 2 + 1, 3] > 0.5:
                image_2 = self._perform_random_zoom(image_2)

            images[0][pair_index, :, :, :] = image_1
            images[1][pair_index, :, :, :] = image_2

        return images

    def get_random_transform_single_image(self, image):
        """ Applies a random augmentation to a single image

            Args:
                image: image to be augmented

            Returns:
                The transformed image
        """

        random_numbers = np.random.random(size=(4))

        if random_numbers[0] > 0.5:
            image = self._perform_random_rotation(image)
        if random_numbers[1] > 0.5:
            image = self._perform_random_shear(image)
        if random_numbers[2] > 0.5:
            image = self._perform_random_shift(image)
        if random_numbers[3] > 0.5:
            image = self._perform_random_zoom(image)

        return image

    def get_random_transform_single_image_transformation(self, image):
        """ Applies a random augmentation to a single image

            Args:
                image: image to be augmented

            Returns:
                The transformed image
        """

        random_numbers = np.random.random(size=(4))
        augmumented_str = ""

        if random_numbers[0] > 0.6:
            image = self._perform_random_rotation_pil(image)
            augmumented_str += "r"
        if random_numbers[1] > 0.6:
            image = self._perform_random_shear_pil(image)
            augmumented_str += "s"
        if random_numbers[2] > 0.6:
            image = self._perform_random_shift_pil(image)
            augmumented_str += "f"
        if random_numbers[3] > 0.6:
            image = self._perform_random_zoom_pil(image)
            augmumented_str += "z"
        # image = np.asarray(image)
        # if random_numbers[4] > 0.6:
        #     image = self._gaussian_blur(image)
        #     augmumented_str += "b"
        # if random_numbers[5] > 0.7:
        #     image = self._adjust_brightness_contrast(image)
        #     augmumented_str += "l"
        # if random_numbers[6] > 0.6 and random_numbers[5] < 0.7:
        #     image = self._adjust_brightness_contrast(image, True)
        #     augmumented_str += "d"
        # if random_numbers[7] > 0.5 and random_numbers[6] < 0.6 and random_numbers[5] < 0.7:
        #     image = self._adjust_brightness_contrast(image, contrast=True)
        #     augmumented_str += "c"

        return image, augmumented_str

    def get_random_transform_single_image_transformation_after_background(self, image, augmumented_str):
        random_numbers = np.random.random(size=(4))
        if random_numbers[0] > 0.6:
            image = self._gaussian_blur(image)
            augmumented_str += "b"
        if random_numbers[1] > 0.7:
            image = self._adjust_brightness_contrast(image)
            augmumented_str += "l"
        if random_numbers[2] > 0.6 and random_numbers[1] < 0.7:
            image = self._adjust_brightness_contrast(image, True)
            augmumented_str += "d"
        if random_numbers[3] > 0.5 and random_numbers[2] < 0.6 and random_numbers[1] < 0.7:
            image = self._adjust_brightness_contrast(image, contrast=True)
            augmumented_str += "c"

        return image, augmumented_str
