import os
import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt


def show_image(image: np.array, cmap: str = "", axis="off") -> None:
    plt.figure(figsize=(8, 8))
    plt.axis(axis)

    if cmap != "":
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)

def create_dir(dir_name: str) -> None:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def preprocess_image(image: np.array, target_size: int = 244) -> np.array:
    image = cv2.resize(image, (target_size, target_size))

    if len(image.shape) > 0:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)

    image /= 255.0

    return image


def get_file_name(file_path: str) -> str:
    name_segments = os.path.basename(file_path).split(".")[:-1]
    name = "".join(f"{x}." for x in name_segments)
    return name.rstrip(".")


def get_random_slice(image: np.array) -> np.array:
    if (len(image.shape)) > 2:
        height, width, _ = image.shape
    else:
        height, width = image.shape

    print(image.shape)
    x_start = random.randint(0, width - height - 1)
    slice = image[:, x_start : x_start + height]
    return slice


def binarize_line(
    img: np.array, adaptive: bool = True, block_size: int = 15, c: int = 13
) -> np.array:
    line_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if adaptive:
        bw = cv2.adaptiveThreshold(
            line_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c,
        )

    else:
        _, bw = cv2.threshold(line_img, 120, 255, cv2.THRESH_BINARY)

    bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    return bw


def resize_image(
    orig_img: np.array, target_width: int = 2048
) -> tuple[np.array, float]:
    if orig_img.shape[1] > orig_img.shape[0]:
        resize_factor = round(target_width / orig_img.shape[1], 2)
        target_height = int(orig_img.shape[0] * resize_factor)

        resized_img = cv2.resize(orig_img, (target_width, target_height))

    else:
        target_height = target_width
        resize_factor = round(target_width / orig_img.shape[0], 2)
        target_width = int(orig_img.shape[1] * resize_factor)
        resized_img = cv2.resize(orig_img, (target_width, target_height))

    return resized_img, resize_factor


def pad_image(
    img: np.array, patch_size: int = 64, is_mask=False, pad_value: int = 255
) -> tuple[np.array, tuple[int, int]]:
    x_pad = (math.ceil(img.shape[1] / patch_size) * patch_size) - img.shape[1]
    y_pad = (math.ceil(img.shape[0] / patch_size) * patch_size) - img.shape[0]

    if is_mask:
        pad_y = np.zeros(shape=(y_pad, img.shape[1], 3), dtype=np.uint8)
        pad_x = np.zeros(shape=(img.shape[0] + y_pad, x_pad, 3), dtype=np.uint8)
    else:
        pad_y = np.ones(shape=(y_pad, img.shape[1], 3), dtype=np.uint8)
        pad_x = np.ones(shape=(img.shape[0] + y_pad, x_pad, 3), dtype=np.uint8)
        pad_y *= pad_value
        pad_x *= pad_value

    img = np.vstack((img, pad_y))
    img = np.hstack((img, pad_x))

    return img, (x_pad, y_pad)


def patch_image(
    img: np.array, patch_size: int = 64, overlap: int = 2, is_mask=False
) -> list:
    """
    A simple slicing function.
    Expects input_image.shape[0] and image.shape[1] % patch_size = 0
    """

    y_steps = img.shape[0] // patch_size
    x_steps = img.shape[1] // patch_size

    patches = []

    for y_step in range(0, y_steps):
        for x_step in range(0, x_steps):
            x_start = x_step * patch_size
            x_end = (x_step * patch_size) + patch_size

            crop_patch = img[
                y_step * patch_size : (y_step * patch_size) + patch_size, x_start:x_end
            ]
            patches.append(crop_patch)

    return patches, y_steps


def unpatch_image(image, pred_patches: list) -> np.array:
    patch_size = pred_patches[0].shape[1]

    x_step = math.ceil(image.shape[1] / patch_size)

    list_chunked = [
        pred_patches[i : i + x_step] for i in range(0, len(pred_patches), x_step)
    ]

    final_out = np.zeros(shape=(1, patch_size * x_step))

    for y_idx in range(0, len(list_chunked)):
        x_stack = list_chunked[y_idx][0]

        for x_idx in range(1, len(list_chunked[y_idx])):
            patch_stack = np.hstack((x_stack, list_chunked[y_idx][x_idx]))
            x_stack = patch_stack

        final_out = np.vstack((final_out, x_stack))

    final_out = final_out[1:, :]
    final_out *= 255

    return final_out


def unpatch_prediction(prediction: np.array, y_splits: int) -> np.array:
    prediction *= 255
    prediction_sliced = np.array_split(prediction, y_splits, axis=0)
    prediction_sliced = [np.concatenate(x, axis=1) for x in prediction_sliced]
    prediction_sliced = np.vstack(np.array(prediction_sliced))

    return prediction_sliced


def rotate_from_angle(image: np.array, angle: float) -> np.array:
    rows, cols = image.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    rotated_img = cv2.warpAffine(image, rot_matrix, (cols, rows), borderValue=(0, 0, 0))

    return rotated_img


def randomly_rotate(image: np.array, limit: int = 10) -> np.array:
    angle_range = random.randint(0, 1)

    if angle_range == 0:
        rotation = random.randint(-limit, limit)

    else:
        rotation = random.randint(180-limit, 180+limit)

    image = rotate_from_angle(image, angle=rotation)

    return image


def calculate_angle(line_prediction: np.array) -> np.array:
    slice = get_random_slice(line_prediction)
    contours, _ = cv2.findContours(slice, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        angles = []
        center, _, angle = cv2.minAreaRect(contour)
        angles.append(angle)

    mean_angle = np.mean(angles)

    if mean_angle > 45:
        target_angle = -(90 - mean_angle)
    else:
        target_angle = mean_angle

    return target_angle
