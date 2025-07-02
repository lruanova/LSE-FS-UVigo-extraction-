from dataclasses import dataclass
from typing import Any, Optional
from keypoint_extraction_pipeline.schemas.keypoints import FrameLandmarks
from pydantic import BaseModel, Field


@dataclass
class Annotation:
    start_time_ms: Optional[float]
    end_time_ms: Optional[float]
    label: str
    video_path: Optional[str] = None


class AnnotationMetadata(BaseModel):
    segment_id: str = Field(
        ..., description="Unique ID for this annotation/segment, e.g., from ELANParser"
    )
    video_path: str = Field(..., description="Path to the video file")
    start_time_ms: Optional[float] = Field(
        ..., description="Annotation start time in milliseconds"
    )
    end_time_ms: Optional[float] = Field(
        ..., description="Annotation end time in milliseconds"
    )
    label: str = Field(..., description="The annotation label (gloss)")
    extractor_name: Optional[str] = Field(
        None, description="Name of the keypoint extractor used"
    )
    signer_id: Optional[str] = None
    handness: Optional[str] = None
    original_video_id: Optional[str] = None
    custom_properties: dict[str, Any] = Field(default_factory=dict)


class AnnotationRecord(BaseModel):
    """Represents a complete annotated segment with its metadata and per-frame keypoints."""

    metadata: AnnotationMetadata
    frames: list[FrameLandmarks] = Field(
        default_factory=list, description="Sequence of frame-level landmark sets"
    )
