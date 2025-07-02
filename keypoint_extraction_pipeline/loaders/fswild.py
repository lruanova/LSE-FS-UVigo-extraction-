import logging
import os
from pathlib import Path
from datasets import (
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    Split,
    SplitGenerator,
    Value,
)
from typing import Any, Dict, Iterator, Tuple
from keypoint_extraction_pipeline.parsers.csv_parser import CSVParser


class FSWildDataset(GeneratorBasedBuilder):
    def _info(self) -> DatasetInfo:
        """Returns the dataset metadata as a DatasetInfo object."""

        return DatasetInfo(
            description="Chicago FSWild dataset..",
            features=Features(
                {
                    "segment_id": Value("string"),  # unique ID per annotation
                    "video_path": Value("string"),  # video path
                    "start_time_ms": Value("float64"),  # segment start (ms)
                    "end_time_ms": Value("float64"),  # segment end (ms)
                    "label": Value("string"),  # label
                }
            ),
        )

    def _split_generators(self, dl_manager: Any) -> list[SplitGenerator]:
        """
        Returns the dataset splits. Expects CSV files in data_dir.
        """
        data_dir = Path(self.config.data_dir).resolve()
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "csv_path": os.path.join(data_dir, "train.csv"),
                    "split_name": "train",
                    "data_dir": data_dir,
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "csv_path": os.path.join(data_dir, "validation.csv"),
                    "split_name": "validation",
                    "data_dir": data_dir,
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "csv_path": os.path.join(data_dir, "test.csv"),
                    "split_name": "test",
                    "data_dir": data_dir,
                },
            ),
        ]

    def _generate_examples(
        self, csv_path: str, split_name: str = "", data_dir: str = ""
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """
        Generates examples from the given CSV file.

        Yields:
            key: segment_id
            example: Dict with segment_id, video_path, start_time_ms, end_time_ms, label
        """
        logging.info(f"Generating examples for split '{split_name}' from {csv_path}")

        parser = CSVParser(
            video_id_field="filename",
            start_time_field="start_time",
            end_time_field="number_of_frames",
            label_field="label_proc",
            videos_prefix=os.path.join(data_dir, split_name),
            process_full_video=True,
        )

        try:
            for idx, annotation in enumerate(parser.parse(csv_path)):
                frames_dir = data_dir / split_name / annotation.video_path
                if not frames_dir.is_dir():
                    print(f"ERROR: {frames_dir} no existe; se omite la muestra")
                    continue

                ann = {
                    "segment_id": f"fs_{split_name}_{idx}",
                    "video_path": str(frames_dir),
                    "start_time_ms": annotation.start_time_ms,
                    "end_time_ms": annotation.end_time_ms,
                    "label": annotation.label,
                }
                yield idx, ann
        except FileNotFoundError as e:
            logging.error(f"CSV file not found: {e}")
            return
