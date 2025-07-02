from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord
from keypoint_extraction_pipeline.schemas.keypoints import Point3D
from keypoint_extraction_pipeline.transformations.operators.base_operator import (
    BaseOperator,
)


class AddWristVelocityOperator(BaseOperator):

    def __init__(self):
        super().__init__(subsets_to_apply=["left_hand", "right_hand"])

    def apply(self, annotation: AnnotationRecord) -> AnnotationRecord:
        prev_left = None
        prev_right = None

        for frame in annotation.frames:
            # >> Left hand
            if frame.left_hand and frame.left_hand.keypoints:
                w_left = frame.left_hand.keypoints[0]  # 0 = wrist
                if (
                    w_left.x is not None
                    and w_left.y is not None
                    and w_left.z is not None
                ):
                    if prev_left is None:
                        vx_l = vy_l = vz_l = 0.0
                    else:
                        vx_l = w_left.x - prev_left.x
                        vy_l = w_left.y - prev_left.y
                        vz_l = w_left.z - prev_left.z
                    frame.left_hand_velocity = Point3D(x=vx_l, y=vy_l, z=vz_l)
                    prev_left = w_left.model_copy()
                else:
                    # Invalid coords = set to 0.0
                    frame.left_hand_velocity = Point3D(x=0.0, y=0.0, z=0.0)
                    prev_left = None
            else:
                # No keypoints = set to 0.0
                frame.left_hand_velocity = Point3D(x=0.0, y=0.0, z=0.0)
                prev_left = None

            # >> Right hand
            if frame.right_hand and frame.right_hand.keypoints:
                w_right = frame.right_hand.keypoints[0]
                if (
                    w_right.x is not None
                    and w_right.y is not None
                    and w_right.z is not None
                ):
                    if prev_right is None:
                        vx_r = vy_r = vz_r = 0.0
                    else:
                        vx_r = w_right.x - prev_right.x
                        vy_r = w_right.y - prev_right.y
                        vz_r = w_right.z - prev_right.z
                    frame.right_hand_velocity = Point3D(x=vx_r, y=vy_r, z=vz_r)
                    prev_right = w_right.model_copy()
                else:
                    frame.right_hand_velocity = Point3D(x=0.0, y=0.0, z=0.0)
                    prev_right = None
            else:
                frame.right_hand_velocity = Point3D(x=0.0, y=0.0, z=0.0)
                prev_right = None

        return annotation
