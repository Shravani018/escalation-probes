"""PCA reference space fitted on single-turn tokens, and centroid comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class ReferenceSpace:
    """PCA fitted on single-turn training tokens, with class centroids in PCA space."""

    pca: PCA
    benign_centroid: np.ndarray
    harmful_centroid: np.ndarray

    @property
    def variance_explained(self) -> np.ndarray:
        """Cumulative explained variance ratio per component."""
        return np.cumsum(self.pca.explained_variance_ratio_)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Project tokens into the reference space."""
        return self.pca.transform(x)

    def centroid_report(self, x: np.ndarray, n_components: int) -> dict:
        """Distance of tokens to each centroid using the top `n_components`.

        Returns mean distances and the share of tokens closer to the harmful centroid.
        """
        z = self.transform(x)[:, :n_components]
        d_benign = np.linalg.norm(z - self.benign_centroid[:n_components], axis=1)
        d_harmful = np.linalg.norm(z - self.harmful_centroid[:n_components], axis=1)
        return {
            "n_components": n_components,
            "mean_dist_benign": float(d_benign.mean()),
            "mean_dist_harmful": float(d_harmful.mean()),
            "share_nearer_harmful": float((d_harmful < d_benign).mean()),
        }


def fit_reference(x_train: np.ndarray, y_train: np.ndarray, n_components: int, seed: int) -> ReferenceSpace:
    """Fit PCA on training tokens only and compute class centroids in PCA space."""
    pca = PCA(n_components=n_components, random_state=seed).fit(x_train)
    z = pca.transform(x_train)
    return ReferenceSpace(pca, z[y_train == 0].mean(axis=0), z[y_train == 1].mean(axis=0))
