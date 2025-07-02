from abc import ABC, abstractmethod
import logging
from pathlib import Path

from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord


class BaseSaver(ABC):
    def __init__(self, save_dir: Path, **kwargs):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def save_record(self, record: AnnotationRecord, filename: str):
        """
        Save a single processed record.
        The record is a SegmentAnnotationData object.
        """
        pass

    @staticmethod
    @abstractmethod
    def load_record(path: str) -> AnnotationRecord:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_file_extension() -> str:
        """Returns the extension of the saver (e.g: .h5, .csv ...) ."""
        return ""

    def __call__(self, batch: dict[str, list[str]]):
        annotations_to_save = batch.get("processed_annotations", [])
        for ann_json in annotations_to_save:
            try:
                record = AnnotationRecord.model_validate_json(ann_json)
                filename = f"{record.metadata.segment_id}{self.get_file_extension()}"
                self.save_record(record, filename)
            except Exception as e:
                logging.error(f"Error saving: {e}", exc_info=True)

        return batch
