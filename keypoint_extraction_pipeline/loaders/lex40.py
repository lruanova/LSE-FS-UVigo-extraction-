import logging
import os
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


class Lex40Dataset(GeneratorBasedBuilder):
    def _info(self) -> DatasetInfo:
        """Returns the dataset metadata as a DatasetInfo object."""

        return DatasetInfo(
            description="Lex40 dataset from Signamed project.",
            features=Features(
                {
                    "segment_id": Value("string"),  # unique ID per annotation
                    "video_path": Value("string"),  # video path
                    "start_time_ms": Value("float64"),  # segment start (ms)
                    "end_time_ms": Value("float64"),  # segment end (ms)
                    "label": Value("string"),  # label
                }
            ),
            supervised_keys=("video_path", "label"),
        )

    def _split_generators(self, dl_manager: Any) -> list[SplitGenerator]:
        """
        Returns the dataset splits. Expects CSV files in data_dir.
        """
        data_dir = self.config.data_dir
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "csv_path": os.path.join(data_dir, "train.csv"),
                    "split_name": "train",
                },
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={
                    "csv_path": os.path.join(data_dir, "validation.csv"),
                    "split_name": "validation",
                },
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "csv_path": os.path.join(data_dir, "test.csv"),
                    "split_name": "test",
                },
            ),
        ]

    def _find_media_files(self, directory: str, valid_extensions: list[str]):
        """Helper to find video files recursively."""
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    yield os.path.join(root, file)

    def _generate_examples(
        self, csv_path: str, split_name: str = ""
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """
        Generates examples from the given CSV file.

        Yields:
            key: segment_id
            example: Dict with segment_id, video_path, start_time_ms, end_time_ms, label
        """
        logging.info(f"Generating examples for split '{split_name}' from {csv_path}")

        parser = CSVParser(
            video_id_field=self.config.video_id_field,
            start_time_field=self.config.start_time_field,
            end_time_field=self.config.end_time_field,
            label_field=self.config.label_field,
            videos_prefix=self.config.videos_prefix,
            process_full_video=self.config.process_full_video,
        )

        try:
            for idx, annotation in enumerate(parser.parse(csv_path)):
                ann_id = f"don_{idx}"
                ann = {
                    "segment_id": f"{split_name}_{idx}",
                    "video_path": annotation.video_path,
                    "start_time_ms": annotation.start_time_ms,
                    "end_time_ms": annotation.end_time_ms,
                    "label": annotation.label,
                }
                yield ann_id, ann
        except FileNotFoundError as e:
            logging.error(f"CSV file not found: {e}")
            return
