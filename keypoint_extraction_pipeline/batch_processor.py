import logging
import os
from typing import Any

import cv2
import numpy as np

from keypoint_extraction_pipeline.schemas.annotation import (
    AnnotationMetadata,
    AnnotationRecord,
)
from keypoint_extraction_pipeline.schemas.keypoints import FrameLandmarks
from omegaconf import DictConfig
from hydra.utils import instantiate

logging.getLogger().setLevel(logging.INFO)


class BatchProcessor:
    """Callable class for ray that orchestrates the parallel processing of batches of data."""

    def __init__(
        self,
        extractor_config: DictConfig,
        frame_extractor_config: DictConfig,
    ):
        """
        Instantiates the keypoint extractor and frame extractor for
        each worker on ray.map_batches() with corresponding configuration.

        Args:
            extractor_config: configuration for the keypoint extractor.
            frame_extractor_config: configuration for the frame extractor.
        """
        logging.info(f"[Worker] inicializado en PID={os.getpid()}")
        self.kp_extractor = instantiate(extractor_config)
        self.frame_extractor = instantiate(frame_extractor_config)
        cv2.setNumThreads(1)

    def __call__(self, batch: dict[str, Any]) -> dict[str, list[dict]]:
        """
        Receives and processes a batch of data (segment_info) and
        returns a batch of AnnotationRecord objects (as dictionaries).

        Args:
            batch: dict with info of each segment.

        """

        logging.info(
            f"[Worker {os.getpid()}] recibidos {len(batch['segment_id'])} items"
        )

        output_annotations: list[dict] = []
        num_items_in_batch = len(batch["segment_id"])

        for i in range(num_items_in_batch):
            # Create AnnotationMetadata
            segment_metadata_dict = {
                "segment_id": batch["segment_id"][i],
                "video_path": batch["video_path"][i],
                "start_time_ms": (
                    batch["start_time_ms"][i] if batch["start_time_ms"][i] else np.nan
                ),
                "end_time_ms": (
                    batch["end_time_ms"][i] if batch["end_time_ms"][i] else np.nan
                ),
                "label": batch["label"][i],
            }
            try:
                metadata = AnnotationMetadata(**segment_metadata_dict)
            except Exception as e:
                logging.error(
                    f"Error creating AnnotationMetadata for segment {batch['segment_id'][i]}: {e}. Skipping."
                )
                continue

            # Get frames (if video_path is directory handles images within, if video extension handles segment)
            frames_iterable = self.frame_extractor.extract_segment(
                video_path=metadata.video_path,
                start_time_ms=metadata.start_time_ms,
                end_time_ms=metadata.end_time_ms,
            )

            # get keypoints
            try:
                frame_results_list: list[FrameLandmarks] = self.kp_extractor.process(
                    frames_iterable
                )
            except Exception as e:
                logging.error(
                    f"Error procesando keypoints para segmento {metadata.segment_id}: {e}",
                    exc_info=True,
                )
                frame_results_list = [self.kp_extractor.empty_frame()]

            if not frame_results_list:  # TODO: HARDCODED, need to modify pyarrow schema
                frame_results_list = [self.kp_extractor.empty_frame()]

            # Create and serialize AnnotationRecord
            annotation_record = AnnotationRecord(
                metadata=metadata, frames=frame_results_list
            )
            output_annotations.append(annotation_record.model_dump_json())
        # ray batch: dict of lists
        return {"processed_annotations": output_annotations}
