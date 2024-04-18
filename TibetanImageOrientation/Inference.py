import os
import cv2
import logging
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from scipy.special import expit
from TibetanImageOrientation.Utils import (
    binarize_line,
    resize_image,
    pad_image,
    patch_image,
    unpatch_image,
    unpatch_prediction,
    get_random_slice,
    preprocess_image,
)


class LineDetection:
    def __init__(
        self,
        model_file="TibetanImageOrientation/Models/line_model_v1_q.onnx",
        class_threshold: int = 0.8,
    ) -> None:
        self._onnx_model_file = model_file
        self._patch_size = 512
        self._class_threshold = class_threshold
        # adjust execution Providers if applicable, see: https://onnxruntime.ai/docs/execution-providers
        self.execution_providers = ["CPUExecutionProvider", "CUDAExecutionProvider"]

        self._init()

    def _init(self) -> None:
        if self._onnx_model_file is not None and os.path.isfile(self._onnx_model_file):
            try:
                self._inference = ort.InferenceSession(
                    self._onnx_model_file, providers=self.execution_providers
                )
                self.can_run = True
                logging.info("LineModel file successfully loaded.")
            except Exception as error:
                logging.error(
                    f"Error loading model file: {error}, file: {self._onnx_model_file}"
                )
                self.can_run = False
        else:
            self.can_run = False
            logging.error(f"No valid model file provided: {self._onnx_model_file}")

    def run(
        self,
        original_image: np.array,
        unpatch_type: int = 0,
        class_threshold: float = 0.7,
    ) -> np.array:
        image, _ = resize_image(original_image)
        padded_img, (pad_x, pad_y) = pad_image(image, self._patch_size)
        image_patches, y_splits = patch_image(padded_img, self._patch_size)
        image_batch = np.array(image_patches)
        image_batch = image_batch.astype(np.float32)
        image_batch /= 255.0

        image_batch = np.transpose(image_batch, axes=[0, 3, 1, 2])  # make B x C x H xW

        ort_batch = ort.OrtValue.ortvalue_from_numpy(image_batch)
        prediction = self._inference.run_with_ort_values(
            ["output"], {"input": ort_batch}
        )
        prediction = prediction[0].numpy()
        prediction = np.squeeze(prediction, axis=1)
        prediction = expit(prediction)
        prediction = np.where(prediction > class_threshold, 1.0, 0.0)
        pred_list = [prediction[x, :, :] for x in range(prediction.shape[0])]

        if unpatch_type == 0:
            unpatched_image = unpatch_image(image, pred_list)
        else:
            unpatched_image = unpatch_prediction(image, y_splits)

        stitched_image = unpatched_image[
            : unpatched_image.shape[0] - pad_y, : unpatched_image.shape[1] - pad_x
        ]
        stitched_image = cv2.resize(
            stitched_image, (original_image.shape[1], original_image.shape[0])
        )
        stitched_image = stitched_image.astype(np.uint8)

        return stitched_image


class FlipDetection:
    def __init__(self, onnx_model_path: str) -> None:
        super().__init__()
        self.onnx_model = onnx_model_path
        self.session = ort.InferenceSession(self.onnx_model)

    def run(self, image: np.array, slice: bool = True):
        
        if slice:
            img = get_random_slice(image)

        else:
            img = image

        t_image = preprocess_image(img)
        t_image = np.expand_dims(t_image, axis=0)
        t_image = np.expand_dims(t_image, axis=-1)

        ort_batch = ort.OrtValue.ortvalue_from_numpy(t_image)
        prediction = self.session.run_with_ort_values(
            ["dense"], {"input_1": ort_batch}
        )
        prediction = prediction[0].numpy()

        is_flipped = round(prediction[0][0], 2)
        is_correct = round(prediction[0][1], 2)

        return img, is_flipped, is_correct
