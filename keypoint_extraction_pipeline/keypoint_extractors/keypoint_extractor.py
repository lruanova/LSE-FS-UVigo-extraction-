from abc import ABC, abstractmethod
from typing import Iterable
import numpy as np
from keypoint_extraction_pipeline.schemas.keypoints import LandmarkSet


class KeypointExtractor(ABC):
    @abstractmethod
    def process(self, frames: Iterable[np.ndarray]) -> list[LandmarkSet]:
        """
        Detects keypoints on a single frame.

        Args:
            frames: ndarray of frames to process.

        Returns:
            A list of DTOs with detected keypoint sets for each frame.
        """
        raise NotImplementedError()
