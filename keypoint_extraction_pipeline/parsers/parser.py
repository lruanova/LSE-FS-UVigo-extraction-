from abc import abstractmethod
from typing import List
from keypoint_extraction_pipeline.schemas.annotation import Annotation


class AnnotationParser:
    def __init__(self, process_full_video):
        self.process_full_video = process_full_video

    @abstractmethod
    def parse(self, source: str) -> List[Annotation]:
        pass

    def ms_to_frames(self, ms: float, fps: float) -> int:
        return int(ms / 1000 * fps)
