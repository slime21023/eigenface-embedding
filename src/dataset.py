import cv2 as cv
import os
import numpy as np


def load(gray=False) -> np.ndarray:
    """
    To load the prepared image dataset to a feature matrix
    """
    _resized_path = "../data/prepare/resized"

    def to_feature(image_name, gray=False) -> np.ndarray:
        file_path = os.path.join(_resized_path, image_name)
        img = cv.imread(file_path)
        mode = cv.COLOR_BGR2GRAY if gray else cv.COLOR_BGR2RGB
        cvt_img = cv.cvtColor(img, mode)
        return cvt_img.flatten()

    features = [to_feature(name, gray) for name in os.listdir(_resized_path)]
    features_mat = np.asarray(features)
    return features_mat


def resize(image: cv.typing.MatLike) -> cv.typing.MatLike:
    """
    Resize the image to the size (w: 90, h: 120)
    and interpolation by `cv.INTER_CUBIC`
    """
    return cv.resize(image, (90, 120), interpolation=cv.INTER_CUBIC)


def load_image(path: os.PathLike, gray=False) -> np.ndarray:
    """
    Load the image by path and resize it to the size (w: 90, h: 120)
    , interpolation by `cv.INTER_CUBIC`
    """

    img = cv.imread(path)
    mode = cv.COLOR_BGR2GRAY if gray else cv.COLOR_BGR2RGB
    cvt_img = cv.cvtColor(img, mode)
    return resize(cvt_img).flatten()
