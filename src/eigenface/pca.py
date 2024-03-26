import src.dataset as dataset
import numpy as np
import os
import h5py
from skl2onnx import convert_sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
from skl2onnx.common.data_types import FloatTensorType
from sklearn.decomposition import PCA


class FacePCA:
    def __init__(self, gray: bool = False, n_components: int = 384, **kwargs):
        """
        Create a sklearn PCA model from the prepared resized face data.

        Args:
        `n_components`: int, default=`384`
        `gray`: bool, set the cv image channels, default=`False`, RGB model

        Returns:
        PCA: the eigenface embedding model based on the sklearn PCA
        """
        self.gray = gray
        features_mat = dataset.load(gray) / 255.0
        self._features_shape = features_mat.shape

        self.pca = PCA(
            n_components=n_components, svd_solver="auto", whiten=True, **kwargs
        )
        self.pca.fit(features_mat)
        self.features_mat = features_mat

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.pca.transform(features)

    def to_onnx(self, path: os.PathLike) -> None:
        """
        Save the model as a onnx file
        """
        initial_types = [("face_features", FloatTensorType([1, None]))]
        pipe = Pipeline([("pca", self.pca)])

        # Convert the model.
        model_onnx = convert_sklearn(pipe, "pca", initial_types=initial_types)

        # save.
        with open(path, "wb") as f:
            f.write(model_onnx.SerializeToString())

    def save_embedding(self, path: os.PathLike, has_normalize: bool = True) -> None:
        with h5py.File(path, "w") as f:
            all_embeddings = self.pca.transform(self.features_mat)
            if has_normalize:
                all_embeddings = normalize(all_embeddings)
            f.create_dataset("embeds", data=all_embeddings)
