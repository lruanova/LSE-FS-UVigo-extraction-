from typing import Optional, List
from pydantic import BaseModel


class Point3D(BaseModel):
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None


class LandmarkSet(BaseModel):
    keypoints: List[Point3D]
    labels: Optional[List[str]] = None
    confidence_scores: Optional[List[float]] = None


class FrameLandmarks(BaseModel):
    pose: Optional[LandmarkSet] = None
    left_hand: Optional[LandmarkSet] = None
    right_hand: Optional[LandmarkSet] = None
    face: Optional[LandmarkSet] = None
    left_hand_velocity: Optional[Point3D] = None
    right_hand_velocity: Optional[Point3D] = None
