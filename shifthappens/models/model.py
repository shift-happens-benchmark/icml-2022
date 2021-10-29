from typing import Tuple
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np


class ConfidenceModelMixin:
    # inherit from this class if your model returns confidences
    pass


class FeaturesModelMixin:
    # inherit from this class if your model returns features
    pass


class LabelModelMixin:
    # inherit from this class if your model returns predicted labels
    pass


ModelResult = Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]


class Model(ABC):
    @abstractmethod
    def predict(self, x: np.ndarray) -> ModelResult:
        """
    :param x: Batch of images
    :returns: Tuple of the predicted labels, predicted class confidences (optional) and predicted image features (optional).
    """
        pass
