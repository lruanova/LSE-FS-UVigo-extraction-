import logging
from math import nan
from typing import Iterable, Optional, Dict, Any
from keypoint_extraction_pipeline.keypoint_extractors.keypoint_extractor import (
    KeypointExtractor,
)
from keypoint_extraction_pipeline.schemas.keypoints import (
    FrameLandmarks,
    LandmarkSet,
    Point3D,
)
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np

SET_NAME_POSE = "pose"
SET_NAME_HAND_LEFT = "left_hand"
SET_NAME_HAND_RIGHT = "right_hand"

NUM_KPS_PER_SET = {
    "pose": 33,
    "left_hand": 21,
    "right_hand": 21,
    "face": 468,
}


class MediapipeHolisticExtractor(KeypointExtractor):
    EXTRACTOR_NAME = "mediapipe_holistic"

    def __init__(self, mediapipe_cfg: Dict[str, Any]):
        self.cfg = mediapipe_cfg
        static_mode = self.cfg.get("static_image_mode", True)
        logging.info(
            f"Initializing MediaPipe Holistic (static_image_mode={static_mode}). Config: {self.cfg}"
        )

        self._mp_model = mp.solutions.holistic.Holistic(  # type: ignore
            static_image_mode=static_mode,
            model_complexity=self.cfg.get("model_complexity", 2),
            smooth_landmarks=self.cfg.get("smooth_landmarks", True) and not static_mode,
            min_detection_confidence=self.cfg.get("min_detection_confidence", 0.5),
            min_tracking_confidence=self.cfg.get("min_tracking_confidence", 0.5),
        )
        logging.info("MediaPipe Holistic Initialized.")

    def _create_landmark_set(
        self, mp_landmarks, num_expected_keypoints: int
    ) -> LandmarkSet:

        if not mp_landmarks or not mp_landmarks.landmark:
            default_point_list = [Point3D() for _ in range(num_expected_keypoints)]
            return LandmarkSet(keypoints=default_point_list)

        return LandmarkSet(
            keypoints=[
                Point3D(x=lm.x, y=lm.y, z=getattr(lm, "z", None))
                for lm in mp_landmarks.landmark
            ]
        )

    def empty_frame(self) -> FrameLandmarks:
        return FrameLandmarks(
            pose=self._create_landmark_set(None, NUM_KPS_PER_SET["pose"]),
            left_hand=self._create_landmark_set(None, NUM_KPS_PER_SET["left_hand"]),
            right_hand=self._create_landmark_set(None, NUM_KPS_PER_SET["right_hand"]),
            face=self._create_landmark_set(None, NUM_KPS_PER_SET["face"]),
        )

    def _create_frame_result(self, mp_res) -> FrameLandmarks:

        frame_data = FrameLandmarks(
            pose=self._create_landmark_set(
                mp_res.pose_landmarks, num_expected_keypoints=NUM_KPS_PER_SET["pose"]
            ),
            left_hand=self._create_landmark_set(
                mp_res.left_hand_landmarks, NUM_KPS_PER_SET["left_hand"]
            ),
            right_hand=self._create_landmark_set(
                mp_res.right_hand_landmarks, NUM_KPS_PER_SET["right_hand"]
            ),
            face=self._create_landmark_set(
                mp_res.face_landmarks, NUM_KPS_PER_SET["face"]
            ),
        )
        return frame_data

    def process(self, frames: Iterable[np.ndarray]) -> list[FrameLandmarks]:
        """
        Processes an iterable of frames and returns a list of FrameLandmarks objects.

        Args:
            frames: ndarray of frames to process. Will be converted to RGB internally.

        """
        out: list[FrameLandmarks] = []

        for idx, frame_rgb in enumerate(frames):
            try:
                mp_res = self._mp_model.process(frame_rgb)
                out.append(self._create_frame_result(mp_res))
            except Exception as e:
                logging.error(f"Error frame {idx}: {e}")
                out.append(self.empty_frame())

        return out

    def __del__(self):
        if hasattr(self, "_mp_model") and hasattr(self._mp_model, "close"):
            logging.info("Closing MediaPipe Holistic instance.")
            self._mp_model.close()
