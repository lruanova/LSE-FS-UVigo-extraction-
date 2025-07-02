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
from typing import Any, Optional

import pympi
from keypoint_extraction_pipeline.parsers.elan_parser import ELANParser


class ESaudeDataset(GeneratorBasedBuilder):
    def _info(self) -> DatasetInfo:
        """Returns the dataset metadata as a DatasetInfo object."""

        return DatasetInfo(
            description="ESaude dataset from Signamed project.",
            features=Features(
                {
                    "segment_id": Value("string"),  # unique ID for each annotation
                    "video_path": Value("string"),  # path to video file
                    "start_time_ms": Value("float64"),  # annotation start time
                    "end_time_ms": Value("float64"),  # annotation end time
                    "label": Value("string"),  # annotation label
                }
            ),
        )

    def _split_generators(self, dl_manager: Any) -> list[SplitGenerator]:
        """
        Returns the dataset splits.

        Args:
            dl_manager (Any): Data manager object

        """

        data_dir: Optional[str] = self.config.data_dir
        if not data_dir:
            raise

        return [
            SplitGenerator(
                name=str(Split.TRAIN),
                gen_kwargs={
                    "videos_dir": os.path.join(data_dir, "train"),
                    "split_name": "TRAIN",
                },
            ),
            SplitGenerator(
                name=str(Split.VALIDATION),
                gen_kwargs={
                    "videos_dir": os.path.join(data_dir, "validation"),
                    "split_name": "VALIDATION",
                },
            ),
            SplitGenerator(
                name=str(Split.TEST),
                gen_kwargs={
                    "videos_dir": os.path.join(data_dir, "test"),
                    "split_name": "TEST",
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
        self, videos_dir: Optional[str] = None, split_name: str = "unknown"
    ):
        """
        Generate examples from the dataset.

        Args:
            videos_dir (str): Path to the directory containing video files

        Yields:
            Tuple: The next example in the dataset
        """
        logging.info(f"Generating examples for ESaude from directory: {videos_dir}.")

        if videos_dir is None:
            raise ValueError("videos_dir must not be None.")

        failures = 0

        parser = ELANParser(
            tier_name="M_Glosa",
            filter_dt=True,
            read_source_from_eaf=False,
            process_full_video=False,
        )

        for video_path in self._find_media_files(videos_dir, valid_extensions=[".mp4"]):
            eaf_path = os.path.splitext(video_path)[0] + ".eaf"

            if not os.path.exists(eaf_path):
                logging.warning(
                    f"EAF file {eaf_path} not found for video {video_path}. Skipping video."
                )
                continue

            # .parse() returns a generator of annotations
            idx = 0

            try:
                for annotation in parser.parse(eaf_path):
                    ann_id = f"esaude_{os.path.basename(eaf_path)}_{idx}"
                    features_dict = {
                        "segment_id": ann_id,
                        "video_path": video_path,
                        "start_time_ms": annotation.start_time_ms,
                        "end_time_ms": annotation.end_time_ms,
                        "label": annotation.label,
                    }

                    yield ann_id, features_dict
                    idx += 1
            except pympi.Elan.EafParseError as e:  # type: ignore
                logging.error(
                    f"Error when parsing {eaf_path}: {e}. Skipping annotation."
                )
                failures += 1
                continue
            except Exception as e:
                logging.error(
                    f"Unexpected error. Skipping annotation. \n Processing: {eaf_path} \
                     Source video : {video_path} \n Exception: {e}",
                    exc_info=True,
                )
                failures += 1
                continue

        logging.info(
            f"📍 Ended processing partition: {split_name} for ESaude dataset . \t Failed annotations: {failures}."
        )
