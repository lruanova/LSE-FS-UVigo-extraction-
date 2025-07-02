import logging
import math
from typing import List, Optional
from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord
from keypoint_extraction_pipeline.transformations.operators.base_operator import (
    BaseOperator,
)
from keypoint_extraction_pipeline.schemas.keypoints import (
    LandmarkSet,
    Point3D,
)
from keypoint_extraction_pipeline.transformations.operators.center_on_origin import (
    CenterOperator,
)

# defaults for mediapipe
DEFAULT_REF_POINT_A_IDX = 0  # Wrist
DEFAULT_REF_POINT_B_IDX = 9  # mid finger MCP


class HandSizeNormalizerOperator(BaseOperator):
    """
    Normalizes the apparent size of detected hands by scaling them. Scaling ensures that the Euclidean distance between two specified reference
    landmarks within the hand (ref_point_a_idx and ref_point_b_idx) matches a target size after normalization.
    This helps mitigate variations in hand size due to camera distance.

    This operator expects that the hand keypoints have ALREADY BEEN CENTERED on (0,0,0). Operates in-place.
    """

    def __init__(
        self,
        fixed_size: float = 0.2,
        ref_point_a_idx: int = DEFAULT_REF_POINT_A_IDX,
        ref_point_b_idx: int = DEFAULT_REF_POINT_B_IDX,
        subsets_to_apply: List[str] = [
            "left_hand",
            "right_hand",
        ],
    ):
        """
        Initializes the HandSizeNormalizer.
        """
        super().__init__(subsets_to_apply)
        if fixed_size <= 0:
            raise ValueError("fixed_size must be positive.")
        if ref_point_a_idx == ref_point_b_idx:
            raise ValueError(
                "ref_point_a_idx and ref_point_b_idx must be different."
            )

        self.fixed_size = fixed_size
        self.ref_point_a_idx = ref_point_a_idx
        self.ref_point_b_idx = ref_point_b_idx

        logging.info(
            f"HandSizeNormalizer initialized with fixed_size: {self.fixed_size}, "
            f"ref_point_a_idx: {self.ref_point_a_idx}, ref_point_b_idx: {self.ref_point_b_idx}, "
            f"applying to: {self.subsets_to_apply}. "
            "Operator expects hands to be pre-centered if scaling should not shift global position."
        )

    @property
    def dependencies(self) -> list[type]:
        return [CenterOperator]

    def apply(self, ann: AnnotationRecord) -> AnnotationRecord:
        """
        Applies hand size normalization to the specified hand subsets in each frame.
        """
        if not self.subsets_to_apply:
            logging.warning(
                "HandSizeNormalizer: No subsets_to_apply configured. Skipping."
            )
            return ann

        for frame_idx, frame_landmarks_data in enumerate(ann.frames):
            for hand_subset_name in self.subsets_to_apply:

                hand_landmark_set: Optional[LandmarkSet] = getattr(
                    frame_landmarks_data, hand_subset_name, None
                )

                if hand_landmark_set and hand_landmark_set.keypoints:
                    hand_landmarks: List[Point3D] = hand_landmark_set.keypoints

                    current_ref_distance = self._get_current_reference_distance(
                        hand_landmarks, hand_subset_name, frame_idx
                    )

                    if (
                        current_ref_distance is not None and current_ref_distance > 1e-6
                    ):  # small epsilon
                        scale_factor = self.fixed_size / current_ref_distance
                        logging.debug(
                            f"Frame {frame_idx}, {hand_subset_name}: "
                            f"Current ref distance (A-B) = {current_ref_distance:.4f}, Target ref distance = {self.fixed_size:.4f}, "
                            f"Scaling factor = {scale_factor:.4f}"
                        )
                        self._scale_hand_keypoints(hand_landmarks, scale_factor)
                    elif (
                        current_ref_distance is not None
                        and current_ref_distance <= 1e-6
                    ):
                        logging.warning(
                            f"Frame {frame_idx}, {hand_subset_name}: Calculated current reference distance "
                            "is zero or near-zero. Skipping scaling for this hand to avoid division by zero "
                            "or extreme scaling factors."
                        )
                else:
                    logging.debug(
                        f"Frame {frame_idx}: No keypoints found for '{hand_subset_name}'. "
                        "Skipping size normalization for this hand."
                    )
        return ann

    def _get_current_reference_distance(
        self, hand_landmarks: List[Point3D], hand_name: str, frame_idx: int
    ) -> Optional[float]:
        """
        Computes the current Euclidean distance between the two specified reference
        landmarks within the hand.

        Args:
            hand_landmarks: Points for the hand.
            hand_name: Name of the hand subset (for logging)
            frame_idx: Index of the current frame (for logging).

        Returns:
            The calculated Euclidean distance between the reference points,
            or None if it cannot be computed.
        """
        max_required_idx = max(self.ref_point_a_idx, self.ref_point_b_idx)
        if not (max_required_idx < len(hand_landmarks)):
            logging.warning(
                f"Frame {frame_idx}, {hand_name}: Reference point A (idx {self.ref_point_a_idx}) or "
                f"B (idx {self.ref_point_b_idx}) index out of bounds for hand landmarks list "
                f"(len {len(hand_landmarks)}). Skipping distance calculation."
            )
            return None

        point_a: Point3D = hand_landmarks[self.ref_point_a_idx]
        point_b: Point3D = hand_landmarks[self.ref_point_b_idx]
        if not (self.is_valid(point_a) and self.is_valid(point_b)):
            return None

        distance = math.sqrt(
            (point_a.x - point_b.x) ** 2
            + (point_a.y - point_b.y) ** 2
            + (point_a.z - point_b.z) ** 2
        )

        return distance

    def _scale_hand_keypoints(self, hand_landmarks: List[Point3D], scale_factor: float):
        """
        Scales all landmarks of a hand by a given factor.
        If the hand was pre-centered (e.g., wrist at (0,0,0)), this scaling occurs
        around that center. Otherwise, it scales with respect to the global origin,
        which will also shift the hand's position.
        """
        for landmark in hand_landmarks:
            if self.is_valid(landmark):
                landmark.x *= scale_factor
                landmark.y *= scale_factor
                landmark.z *= scale_factor
