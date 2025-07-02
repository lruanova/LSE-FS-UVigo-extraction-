import json
import logging
from pathlib import Path
from keypoint_extraction_pipeline.savers.saver import BaseSaver
from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord


class JSONSaver(BaseSaver):
    @staticmethod
    def get_file_extension() -> str:
        return ".json"

    def save_record(self, record: AnnotationRecord, filename: str):
        print(f"JSONSaver: Saving record {record.metadata.segment_id} to {filename}.")
        file_path = self.save_dir / filename
        with open(file_path, "w") as f:
            f.write(record.model_dump_json(indent=2))
        logging.debug(f"Saved annotation to {file_path}")

    @staticmethod
    def load_record(file_path: Path) -> AnnotationRecord:
        with open(file_path, "r") as f:
            data = json.load(f)
            return AnnotationRecord.model_validate(data)
