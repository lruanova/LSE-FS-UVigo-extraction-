import logging
from pathlib import Path
import hydra
from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord
from omegaconf import DictConfig
from keypoint_extraction_pipeline.savers.saver import BaseSaver


class PipelineProcessor:
    """
    Receives batches, applies transforms and returns a dict ready for the saver.
    """

    def __init__(self, pipeline_config: DictConfig, saver_class: BaseSaver):
        self.pipeline = hydra.utils.instantiate(pipeline_config.obj)
        self.saver_class = saver_class

    def __call__(self, batch: dict[str, list[str]]) -> dict[str, list[str]]:
        """
        Loads AnnotationRecord objects from file paths, applies transformations to their
        frame data, and returns the modified AnnotationRecord objects (as dicts).
        """
        output_annotations: list[str] = []

        for file_path_str in batch["file_path"]:
            file_path = Path(file_path_str)
            logging.info(f"[PipelineProcessor] Processing: {file_path}")

            # Load record
            record: AnnotationRecord = self.saver_class.load_record(str(file_path))

            # Apply transformations if frames
            if not record.frames:
                logging.warning(
                    f"{record.metadata.segment_id} has no frames — skipping transforms."
                )
            else:
                self.pipeline(record)

            # Serialize to json
            output_annotations.append(record.model_dump_json())

        return {"processed_annotations": output_annotations}
