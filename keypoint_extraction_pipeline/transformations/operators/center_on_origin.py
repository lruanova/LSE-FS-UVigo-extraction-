import logging
from typing import List
from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord
from keypoint_extraction_pipeline.transformations.operators.base_operator import (
    BaseOperator,
)
from keypoint_extraction_pipeline.schemas.keypoints import (
    FrameLandmarks,
    LandmarkSet,
    Point3D,
)


class CenterOperator(BaseOperator):
    def __init__(self, subsets_to_apply: List[str] = ["left_hand", "right_hand"]):
        super().__init__(subsets_to_apply)

    def apply(self, annotation: AnnotationRecord) -> AnnotationRecord:
        frame_landmark_list: List[FrameLandmarks] = annotation.frames

        if not frame_landmark_list:
            return annotation

        for frame_landmarks_data in frame_landmark_list:

            for subset_name_to_center in self.subsets_to_apply:
                landmark_set_to_process: LandmarkSet | None = getattr(
                    frame_landmarks_data, subset_name_to_center, None
                )

                if landmark_set_to_process:
                    keypoints_as_points: List[Point3D] | None = (
                        landmark_set_to_process.keypoints
                    )

                    if keypoints_as_points and len(keypoints_as_points) > 0:
                        self.__center_hand(keypoints_as_points, wrist_idx=0)
                    else:
                        logging.debug(
                            f"Subset '{subset_name_to_center}' in FrameLandmarks "
                            f"found but has no 'keypoints' (or keypoints list is empty). Skipping centering."
                        )
                else:
                    logging.debug(
                        f"Subset '{subset_name_to_center}' not found as an attribute of FrameLandmarks. "
                        f"Skipping centering for this subset."
                    )
        return annotation

    def __center_hand(self, hand_kps_points: List[Point3D], wrist_idx: int = 0):
        if not hand_kps_points or wrist_idx >= len(hand_kps_points):
            logging.warning(
                f"Hand keypoints list (List[Point3D]) is empty or wrist_idx ({wrist_idx}) is out of bounds. Skipping centering."
            )
            return

        wrist_coords_reference: Point3D = hand_kps_points[wrist_idx].model_copy()

        if not self.is_valid(wrist_coords_reference):
            logging.debug("Wrist coordinates are None - skipping centering.")
            return

        logging.debug(
            f"Centering around original wrist coordinates: X={wrist_coords_reference.x}, Y={wrist_coords_reference.y}, Z={wrist_coords_reference.z}"
        )

        for landmark_point in hand_kps_points:
            if self.is_valid(landmark_point):
                landmark_point.x -= wrist_coords_reference.x
                landmark_point.y -= wrist_coords_reference.y
                landmark_point.z -= wrist_coords_reference.z
