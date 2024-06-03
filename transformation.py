from PIL import Image
from image_augmentor import ImageAugmentor
import math
import numpy as np
import cv2

image = 'result_trnsp.png'

# image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
image = Image.open(image)
# image = np.asarray(image)

rotation_range = [-15, 15]
shear_range = [-0.2 * 180 / math.pi, 0.3 * 180 / math.pi]
zoom_range = [0.8, 1.2]
shift_range = [0.3, 0.3]

augmentor = ImageAugmentor(1, shear_range, rotation_range, shift_range, zoom_range)

# rot_img = augmentor._perform_random_rotation_cv(image)
# cv2.imwrite('rotated_trnsp2.png', rot_img)
for i in range(20,30):
    zoom_img = augmentor._perform_random_zoom_pil(image)
    zoom_img.save(f'zoom-trnsp{i}.png')
    # shift_img = augmentor._perform_random_shift_pil(image)
    # shift_img.save(f'shift_trnsp{i}.png')
    # rot_img = augmentor._perform_random_rotation_pil(image)
    # rot_img.save(f'rotated_trnsp{i}.png')
    # cv2.imwrite(f'rotated_trnsp{i}.png', rot_img)
    # shear_img = augmentor._perform_random_shear_pil(image)
    # shear_img.save(f'shear_trnsp{i}.png')
    # cv2.imwrite(f'shear_trnsp{i}.png', shear_img)

# bright_img = augmentor._adjust_brightness_contrast(image, brightness_factor=0.3, contrast_factor=0.3)
# bright_img = Image.fromarray(bright_img)
# bright_img.save("transformed_images/dark_img4.png")

# blurred_img = augmentor._gaussian_blur(image)
# blurred_img = Image.fromarray(blurred_img)
# blurred_img.save("transformed_images/blurred_img3.png")

# fog_img = augmentor._add_fog_effect(image)
# fog_img = Image.fromarray(fog_img)
# fog_img.save("transformed_images/fog_img12.png")

# fog_img = augmentor._add_fog_effect(image)
# fog_img = Image.fromarray(fog_img)
# fog_img.save("transformed_images/whiter_img3.png")

# rot_img = augmentor._perform_random_rotation(image)
# rot_img = Image.fromarray(rot_img)
# rot_img.save("transformed_images/rot_img.png")

# shear_img = augmentor._perform_random_shear(image)
# shear_img = Image.fromarray(shear_img)
# shear_img.save("transformed_images/shear_img.png")

# shift_img = augmentor._perform_random_shift(image)
# shift_img = Image.fromarray(shift_img)
# shift_img.save("transformed_images/shift_img.png")

# zoom_img = augmentor._perform_random_zoom(image)
# zoom_img = Image.fromarray(zoom_img)
# zoom_img.save("transformed_images/zoom_img424.png")
