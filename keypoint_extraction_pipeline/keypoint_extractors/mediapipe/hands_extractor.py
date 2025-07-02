import logging
from math import nan
from typing import Iterable, List, Optional, Dict, Any
import os

from keypoint_extraction_pipeline.keypoint_extractors.keypoint_extractor import (
    KeypointExtractor,
)
from keypoint_extraction_pipeline.schemas.keypoints import (
    FrameLandmarks,
    LandmarkSet,
    Point3D,
)

import mediapipe as mp
from mediapipe.tasks.python import vision

import numpy as np

SET_NAME_HAND_LEFT = "left_hand"
SET_NAME_HAND_RIGHT = "right_hand"


class MediapipeHandsExtractor(KeypointExtractor):
    """Extract keypoints of hands using Mediapipe Hand Landmarker Task API (Image mode)."""

    EXTRACTOR_NAME = "mediapipe_hands"

    def __init__(self, hands_cfg: Dict[str, Any]):
        """
        Initializes hand landmarker model. Only IMAGE mode for now.

        Args:
            hands_cfg : Specific hand landmarker config.
        """
        self.cfg = hands_cfg
        logging.info(
            f"Initializing MediaPipe HandLandmarker (IMAGE mode). Config: {self.cfg}"
        )

        self.model_path = self.cfg.get("model_path")
        if not os.path.exists(str(self.model_path)):
            raise FileNotFoundError(
                f"Hand landmark model not found at path: {self.model_path}"
            )

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE,  # TODO: Support for VIDEO and LIVESTREAM
            num_hands=self.cfg.get("num_hands", 2),
            min_hand_detection_confidence=self.cfg.get(
                "min_hand_detection_confidence", 0.5
            ),
            min_hand_presence_confidence=self.cfg.get(
                "min_hand_presence_confidence", 0.5
            ),
        )

        try:
            self.landmarker = vision.HandLandmarker.create_from_options(options)
            logging.info("MediaPipe HandLandmarker Initialized successfully.")
        except Exception as e:
            logging.error(
                f"Failed to initialize MediaPipe HandLandmarker: {e}", exc_info=True
            )
            raise e

    def _create_landmark_set(self, mp_landmarks) -> Optional[LandmarkSet]:
        if not mp_landmarks or not mp_landmarks.landmark:
            return LandmarkSet(keypoints=[Point3D(x=nan, y=nan, z=nan)])
        else:
            landmark_list = [
                Point3D(x=lm.x, y=lm.y, z=getattr(lm, "z", nan))
                for lm in mp_landmarks.landmark
            ]
            return LandmarkSet(keypoints=landmark_list)

    def _create_frame_result(
        self, hand_result
    ) -> FrameLandmarks:
        """Creates FrameLandmarks DTO from MediaPipe hands results."""
        frame_data = FrameLandmarks()

        if hand_result.hand_landmarks:
            for hand_index in range(len(hand_result.hand_landmarks)):
                # handedness (left/right)
                handedness_categories = hand_result.handedness[hand_index]
                hand_label = handedness_categories[
                    0
                ].category_name.lower()  # left/right
                landmarks_for_hand = hand_result.hand_landmarks[hand_index]

                landmark_set_for_hand = self._create_landmark_set(landmarks_for_hand)

                if landmark_set_for_hand:
                    if hand_label == "left":
                        frame_data.left_hand = landmark_set_for_hand
                    elif hand_label == "right":
                        frame_data.right_hand = landmark_set_for_hand
        return frame_data

    def process(
        self,
        frames: Iterable[np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[FrameLandmarks]:
        """
        Process an iterable of frames and extracts keypoints.

        Args:
            frames: iterable of rgb frames
        """
        results_list: List[FrameLandmarks] = []

        for idx, frame in enumerate(frames):
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                hand_landmarker_result = self.landmarker.detect(mp_image)

                frame_result_dto = self._create_frame_result(hand_landmarker_result)
                results_list.append(frame_result_dto)

            except Exception as e:
                logging.error(
                    f"Error processing frame {idx}. Skipping frame. Exception: {e}",
                    exc_info=True,
                )
                results_list.append(FrameLandmarks())
                continue
        return results_list

    def __del__(self):
        if hasattr(self, "landmarker") and hasattr(self.landmarker, "close"):
            logging.info("Closing MediaPipe HandLandmarker instance.")
            self.landmarker.close()
