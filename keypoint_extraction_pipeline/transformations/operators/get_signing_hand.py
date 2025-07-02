import logging
import math
from keypoint_extraction_pipeline.transformations.operators.center_on_origin import (
    CenterOperator,
)
import numpy as np
from typing import List, Optional, Type, Dict

from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord
from keypoint_extraction_pipeline.transformations.operators.base_operator import (
    BaseOperator,
)
from keypoint_extraction_pipeline.schemas.keypoints import (
    FrameLandmarks,
    LandmarkSet,
    Point3D,
)

SIGNING_HAND_STRATEGIES = ["tallest_hand", "nearest2face", "inter_finger_movement"]

# defaults for MediaPipe
DEFAULT_POSE_LANDMARK_MAP = {
    "nose": 0,
    "left_wrist": 15,
    "right_wrist": 16,
}
DEFAULT_HAND_FINGER_INDICES = list(range(5, 21))  # excluding wrist and base


class GetSigningHandOperator(BaseOperator):
    """
    Operator that determines the signing hand based on a specified strategy and updates the
    <handness> field in the annotation metadata. Requires specific keypoint sets
    (e.g., "pose", "hand_left", "hand_right") to be present in the AnnotationRecord.
    """

    def __init__(
        self,
        strategy: str = "tallest_hand",
        empty_default: str = "right",
        smoothing: Optional[str] = None,
        smoothing_param: float = 0.1,
        pose_landmark_map: Optional[Dict[str, int]] = None,
        hand_finger_indices: Optional[List[int]] = None,
        subsets_to_apply: List[str] = ["pose", "hand_left", "hand_right"],
    ):
        """
        Initializes the GetSigningHandOperator.

        Args:
            strategy: The strategy to use for determining the signing hand.
                      Available strategies: ['tallest_hand', 'nearest2face', 'inter_finger_movement'].
            empty_default: The default hand to assign if a signing hand cannot be determined
            smoothing: Optional smoothing method.
            smoothing_param: Parameter for the smoothing method (e.g., alpha for exponential smoothing).
            pose_landmark_map: A dictionary mapping semantic pose landmark names to their corresponding indices in the pose keypoints list. Defaults to mediapipe.
            hand_finger_indices: A list of indices to be considered as finger landmarks for hand
                                 motion analysis. Defaults to MediaPipe.
            subsets_to_apply: List of keypoint subset names this operator might interact with.
        """
        super().__init__(subsets_to_apply)
        if strategy not in SIGNING_HAND_STRATEGIES:
            raise ValueError(
                f"Invalid signing hand strategy. Must be one of {SIGNING_HAND_STRATEGIES}"
            )
        self.strategy = strategy
        self.empty_default = empty_default
        self.smoothing = smoothing
        self.smoothing_param = smoothing_param

        self.pose_map = (
            pose_landmark_map
            if pose_landmark_map is not None
            else DEFAULT_POSE_LANDMARK_MAP.copy()
        )
        self.hand_finger_indices = (
            hand_finger_indices
            if hand_finger_indices is not None
            else DEFAULT_HAND_FINGER_INDICES[:]
        )

        required_pose_landmarks_for_any_strategy = ["left_wrist", "right_wrist"]
        if self.strategy == "nearest2face":
            required_pose_landmarks_for_any_strategy.append("nose")

        for lm_name in set(
            required_pose_landmarks_for_any_strategy
        ):  # set() to avoid duplicate checks
            if lm_name not in self.pose_map:
                raise ValueError(
                    f"'{lm_name}' not found in pose_landmark_map. It is required for the selected strategy '{self.strategy}' or general operation."
                )

        logging.info(
            f"GetSigningHandOperator initialized with strategy: {self.strategy}, "
            f"empty_default: {self.empty_default}, pose_map: {self.pose_map}, "
            f"hand_finger_indices_count:{len(self.hand_finger_indices)}"
        )

    @property
    def dependencies(self) -> list[Type[BaseOperator]]:
        if self.strategy == "inter_finger_movement":
            return [CenterOperator]
        return []

    @staticmethod
    def available_strategies() -> List[str]:
        """
        Returns a list of available strategies for determining the signing hand.
        """
        return SIGNING_HAND_STRATEGIES

    def apply(self, ann: AnnotationRecord) -> AnnotationRecord:
        """
        Applies the chosen strategy to determine the signing hand and updates
        the <handness> field in the annotation's metadata.

        Args:
            ann: The SegmentAnnotationData to process.

        Returns:
            The processed AnnotationRecord with updated metadata.handness.
        """
        if not ann.frames:
            logging.warning(
                f"Annotation {ann.metadata.segment_id} has no frames. "
                f"Setting signing_hand to default '{self.empty_default}'."
            )
            ann.metadata.handness = self.empty_default
            return ann

        signing_hand: str = self.empty_default

        if self.strategy == "tallest_hand":
            signing_hand = self.get_tallest_hand(ann)
        elif self.strategy == "nearest2face":
            signing_hand = self.get_nearest_to_face(ann)
        elif self.strategy == "inter_finger_movement":
            signing_hand = self.get_inter_finger_movement_hand(ann)
        else:
            raise ValueError(
                f"Unsupported strategy: {self.strategy}"
            )  # should be caught by init

        ann.metadata.handness = signing_hand
        logging.debug(
            f"Annotation {ann.metadata.segment_id}: determined signing hand '{signing_hand}' "
            f"using strategy '{self.strategy}'."
        )
        return ann

    def get_tallest_hand(self, ann: AnnotationRecord) -> str:
        """
        Determines the signing hand based on which wrist has a lower average Y-coordinate
        (higher in the image) across all frames, using pose landmarks.

        Args:
            ann: The SegmentAnnotationData containing frame data with pose landmarks.

        Returns:
            "left" or "right" indicating the signing hand.
        """
        all_right_wrist_y: List[float] = []
        all_left_wrist_y: List[float] = []

        try:
            right_wrist_idx = self.pose_map["right_wrist"]
            left_wrist_idx = self.pose_map["left_wrist"]
        except KeyError as e:
            logging.error(
                f"Missing required pose landmark in pose_map for get_tallest_hand: {e}"
            )
            return self.empty_default

        for frame_data in ann.frames:
            pose_landmark_set: Optional[LandmarkSet] = (
                frame_data.pose
            )  # KPDict >> LandmarkSet
            if pose_landmark_set and pose_landmark_set.keypoints:
                pose_landmarks: List[Point3D] = pose_landmark_set.keypoints
                if len(pose_landmarks) > right_wrist_idx:
                    if self.is_valid(pose_landmarks[right_wrist_idx]):
                        all_right_wrist_y.append(pose_landmarks[right_wrist_idx].y)
                if len(pose_landmarks) > left_wrist_idx:
                    if self.is_valid(pose_landmarks[left_wrist_idx]):
                        all_left_wrist_y.append(pose_landmarks[left_wrist_idx].y)

        if not all_right_wrist_y and not all_left_wrist_y:
            logging.warning(
                f"No pose wrist landmarks found for {ann.metadata.segment_id}. Defaulting hand."
            )
            return self.empty_default

        avg_right_y = (
            sum(all_right_wrist_y) / len(all_right_wrist_y)
            if all_right_wrist_y
            else float("inf")
        )
        avg_left_y = (
            sum(all_left_wrist_y) / len(all_left_wrist_y)
            if all_left_wrist_y
            else float("inf")
        )

        if avg_right_y == float("inf") and avg_left_y == float("inf"):
            return self.empty_default

        return "right" if avg_right_y < avg_left_y else "left"

    def get_nearest_to_face(self, ann: AnnotationRecord) -> str:
        """
        Determines the signing hand based on which wrist is, on average, closer
        to a facial reference point across all frames.

        Args:
            ann: The SegmentAnnotationData containing frame data with pose landmarks.

        Returns:
            "left" or "right" indicating the signing hand.
        """
        all_right_wrist_distances: List[float] = []
        all_left_wrist_distances: List[float] = []

        try:
            nose_idx = self.pose_map["nose"]
            right_wrist_idx = self.pose_map["right_wrist"]
            left_wrist_idx = self.pose_map["left_wrist"]
        except KeyError as e:
            logging.error(
                f"Missing required pose landmark in pose_map for get_nearest_to_face: {e}"
            )
            return self.empty_default

        for frame_data in ann.frames:
            pose_landmark_set: Optional[LandmarkSet] = frame_data.pose
            if pose_landmark_set and pose_landmark_set.keypoints:
                pose_landmarks: List[Point3D] = pose_landmark_set.keypoints
                if len(pose_landmarks) > max(right_wrist_idx, left_wrist_idx, nose_idx):
                    face_center: Point3D = pose_landmarks[nose_idx]
                    right_wrist: Point3D = pose_landmarks[right_wrist_idx]
                    dist_right = math.sqrt(
                        (right_wrist.x - face_center.x) ** 2
                        + (right_wrist.y - face_center.y) ** 2
                        + (right_wrist.z - face_center.z) ** 2
                    )
                    all_right_wrist_distances.append(dist_right)

                    left_wrist: Point3D = pose_landmarks[left_wrist_idx]
                    dist_left = math.sqrt(
                        (left_wrist.x - face_center.x) ** 2
                        + (left_wrist.y - face_center.y) ** 2
                        + (left_wrist.z - face_center.z) ** 2
                    )
                    all_left_wrist_distances.append(dist_left)

        if not all_right_wrist_distances and not all_left_wrist_distances:
            logging.warning(
                f"No pose wrist/face landmarks found for {ann.metadata.segment_id}. Defaulting hand."
            )
            return self.empty_default

        avg_right_dist = (
            sum(all_right_wrist_distances) / len(all_right_wrist_distances)
            if all_right_wrist_distances
            else float("inf")
        )
        avg_left_dist = (
            sum(all_left_wrist_distances) / len(all_left_wrist_distances)
            if all_left_wrist_distances
            else float("inf")
        )

        if avg_right_dist == float("inf") and avg_left_dist == float("inf"):
            return self.empty_default

        return "right" if avg_right_dist < avg_left_dist else "left"

    def get_inter_finger_movement_hand(self, ann: AnnotationRecord) -> str:
        """
        Determines the signing hand based on the cumulative inter-finger movement
        (dispersion of finger displacement vectors) for each hand across the sequence of frames.
        A higher cumulative motion metric suggests more articulated finger movement.

        Args:
            ann: The SegmentAnnotationData containing frame data with hand landmarks.

        Returns:
            "left" or "right" indicating the signing hand.
        """
        if len(ann.frames) < 2:
            logging.warning(
                f"Not enough frames ({len(ann.frames)}) for inter_finger_movement strategy "
                f"for {ann.metadata.segment_id}. Defaulting hand."
            )
            return self.empty_default

        right_motion_per_frame: List[float] = []
        left_motion_per_frame: List[float] = []

        for i in range(1, len(ann.frames)):
            prev_frame_landmarks: FrameLandmarks = ann.frames[i - 1]
            curr_frame_landmarks: FrameLandmarks = ann.frames[i]

            right_motion_per_frame.append(
                self._calculate_frame_motion(
                    (
                        prev_frame_landmarks.right_hand or LandmarkSet(keypoints=[])
                    ).keypoints,
                    (
                        curr_frame_landmarks.right_hand or LandmarkSet(keypoints=[])
                    ).keypoints,
                    self.hand_finger_indices,
                )
            )

            left_motion_per_frame.append(
                self._calculate_frame_motion(
                    (
                        prev_frame_landmarks.left_hand or LandmarkSet(keypoints=[])
                    ).keypoints,
                    (
                        curr_frame_landmarks.left_hand or LandmarkSet(keypoints=[])
                    ).keypoints,
                    self.hand_finger_indices,
                )
            )

        if self.smoothing == "exp_smooth" and self.smoothing_param is not None:
            if right_motion_per_frame:
                right_motion_per_frame = self._exponential_smoothing(
                    right_motion_per_frame, self.smoothing_param
                )
            if left_motion_per_frame:
                left_motion_per_frame = self._exponential_smoothing(
                    left_motion_per_frame, self.smoothing_param
                )

        total_right_motion = sum(right_motion_per_frame)
        total_left_motion = sum(left_motion_per_frame)

        logging.debug(
            f"Inter-finger movement for {ann.metadata.segment_id}: Right={total_right_motion}, Left={total_left_motion}"
        )

        if (
            total_right_motion == 0 and total_left_motion == 0
        ):  # check if both are zero after smoothing
            logging.warning(
                f"No hand movement detected for inter_finger_movement for {ann.metadata.segment_id}. Defaulting hand."
            )
            return self.empty_default

        return "right" if total_right_motion > total_left_motion else "left"

    def _calculate_frame_motion(
        self,
        prev_hand_landmarks: List[Point3D],
        curr_hand_landmarks: List[Point3D],
        keypoints_indices_to_use: List[int],
    ) -> float:
        """
        Calculates a motion metric for a single hand between two consecutive frames,
        based on the sum of pairwise distances between the displacement vectors of specified finger landmarks.
        A higher value indicates more diverse or independent movement among the fingers.
        """
        if not prev_hand_landmarks or not curr_hand_landmarks:
            return 0.0

        prev_coords_list = []
        for idx in keypoints_indices_to_use:
            if idx < len(prev_hand_landmarks) and self.is_valid(
                prev_hand_landmarks[idx]
            ):
                lm = prev_hand_landmarks[idx]
                prev_coords_list.append([lm.x, lm.y, lm.z])

        curr_coords_list = []
        for idx in keypoints_indices_to_use:
            if idx < len(curr_hand_landmarks) and self.is_valid(
                curr_hand_landmarks[idx]
            ):
                lm = curr_hand_landmarks[idx]
                curr_coords_list.append([lm.x, lm.y, lm.z])

        if (
            not prev_coords_list
            or not curr_coords_list
            or len(prev_coords_list) != len(curr_coords_list)
        ):
            logging.debug(
                "_calculate_frame_motion: Not enough valid or matching keypoints after filtering."
            )
            return 0.0

        if len(curr_coords_list) < 2:
            logging.debug(
                f"_calculate_frame_motion: Less than 2 keypoints ({len(curr_coords_list)}) to use for pairwise calculation."
            )
            return 0.0

        prev_positions = np.array(prev_coords_list)
        curr_positions = np.array(curr_coords_list)
        diffs = curr_positions - prev_positions
        pairwise_diff_magnitudes = np.linalg.norm(
            diffs[:, None, :] - diffs[None, :, :], axis=-1
        )

        return pairwise_diff_magnitudes.sum()

    def _exponential_smoothing(self, data: List[float], alpha: float) -> List[float]:
        """
        Applies exponential smoothing to a time series of data.

        Args:
            data: The list of float values representing the time series.
            alpha: The smoothing factor (0 < alpha <= 1). Higher alpha gives more weight to recent data.

        Returns:
            New list containing the smoothed data.
        """
        if not data:
            return []
        if not (0 < alpha <= 1):
            logging.warning(
                f"Exponential smoothing alpha {alpha} out of range (0, 1]. Using raw data."
            )
            return data

        smoothed = [data[0]]
        for t in range(1, len(data)):
            smoothed_val = alpha * data[t] + (1 - alpha) * smoothed[-1]
            smoothed.append(smoothed_val)
        return smoothed
