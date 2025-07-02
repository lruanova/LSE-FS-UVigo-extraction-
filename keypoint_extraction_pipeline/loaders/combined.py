import logging
import os
import csv
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, List, Set

import datasets

from keypoint_extraction_pipeline.parsers.elan_parser import ELANParser
from keypoint_extraction_pipeline.parsers.csv_parser import CSVParser
from keypoint_extraction_pipeline.schemas.annotation import Annotation

logger = logging.getLogger(__name__)


def _parse_annotations_esaude(
    dataset_root_path: Path, dataset_name: str, person_folder_name: str
) -> Iterator[Annotation]:
    person_folder_path = dataset_root_path / dataset_name / person_folder_name
    parser = ELANParser(
        tier_name="M_Glosa",
        filter_dt=True,
        read_source_from_eaf=False,
        process_full_video=False,
    )
    video_extensions = [".mp4"]
    for video_file_path in person_folder_path.rglob("*"):
        if (
            video_file_path.is_file()
            and video_file_path.suffix.lower() in video_extensions
        ):
            eaf_file_path = video_file_path.with_suffix(".eaf")
            if not eaf_file_path.exists():
                logger.warning(
                    f"eSaude: EAF {eaf_file_path} not found for {video_file_path}. Skipping."
                )
                continue
            try:
                yield from parser.parse(str(video_file_path))
            except Exception as e:
                logger.error(
                    f"eSaude: Error parsing ELAN for {video_file_path}: {e}",
                    exc_info=True,
                )


def _parse_annotations_lex40(
    dataset_root_path: Path, dataset_name: str, person_folder_name: str
) -> Iterator[Annotation]:
    sub_dataset_path = dataset_root_path / dataset_name
    annotation_file = sub_dataset_path / f"{person_folder_name}.csv"
    if not annotation_file.exists():
        logger.warning(
            f"Lex40: Annotation CSV {annotation_file} not found. Skipping person {person_folder_name}."
        )
        return

    # videos_prefix is the dataset folder, as paths in CSV (output_filename) include person prefix (e.g: p1/video.mp4).
    videos_prefix_val = str(sub_dataset_path.resolve())
    parser = CSVParser(
        video_id_field="output_filename",
        start_time_field="start_frame",
        end_time_field="end_frame",
        label_field="label",
        videos_prefix=videos_prefix_val,
        process_full_video=True,
    )
    try:
        yield from parser.parse(str(annotation_file))
    except Exception as e:
        logger.error(f"Lex40: Error parsing CSV {annotation_file}: {e}", exc_info=True)


def _parse_annotations_donaciones(
    dataset_root_path: Path, dataset_name: str, person_folder_name: str
) -> Iterator[Annotation]:
    sub_dataset_path = dataset_root_path / dataset_name
    annotation_file = sub_dataset_path / f"{person_folder_name}.csv"
    if not annotation_file.exists():
        logger.warning(
            f"Donaciones: Annotation CSV {annotation_file} not found. Skipping person {person_folder_name}."
        )
        return

    videos_prefix_val = str(sub_dataset_path.resolve())
    parser = CSVParser(
        video_id_field="FullPath",
        start_time_field="start_frame",
        end_time_field="end_frame",
        label_field="Lemma",
        videos_prefix=videos_prefix_val,
        process_full_video=True,
    )
    try:
        yield from parser.parse(str(annotation_file))
    except Exception as e:
        logger.error(
            f"Donaciones: Error parsing CSV {annotation_file}: {e}", exc_info=True
        )


def _parse_annotations_urxencias(
    dataset_root_path: Path, dataset_name: str, person_folder_name: str
) -> Iterator[Annotation]:
    sub_dataset_path = dataset_root_path / dataset_name
    annotation_file = sub_dataset_path / f"{person_folder_name}.csv"
    if not annotation_file.exists():
        logger.warning(
            f"Urxencias: Annotation CSV {annotation_file} not found. Skipping person {person_folder_name}."
        )
        return

    videos_prefix_val = str(sub_dataset_path.resolve())
    parser = CSVParser(
        video_id_field="filename",
        start_time_field="start_time",
        end_time_field="end_time",
        label_field="gloss",
        videos_prefix=videos_prefix_val,
        process_full_video=True,
    )
    try:
        yield from parser.parse(str(annotation_file))
    except Exception as e:
        logger.error(
            f"Urxencias: Error parsing CSV {annotation_file}: {e}", exc_info=True
        )


DATASET_PARSERS_MAP = {
    "esaude": {"func": _parse_annotations_esaude, "person_prefix": "p"},
    "lex40": {"func": _parse_annotations_lex40, "person_prefix": "p"},
    "donaciones": {"func": _parse_annotations_donaciones, "person_prefix": "p"},
    "urxencias": {"func": _parse_annotations_urxencias, "person_prefix": "p"},
}


class CombinedDatasetBuilder(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    SIGNER_META_PATH = "" # path to a csv mapping each signer with corresponding partition

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description="Aggregated dataset from multiple sources parsed with dedicated functions.",
            features=datasets.Features(
                {
                    "segment_id": datasets.Value("string"),
                    "video_path": datasets.Value("string"),
                    "start_time_ms": datasets.Value("float64"),
                    "end_time_ms": datasets.Value("float64"),
                    "label": datasets.Value("string"),
                    "dataset_source": datasets.Value("string"),
                    "person_id": datasets.Value("string"),
                }
            ),
        )

    def _load_signer_splits(self, signer_meta_file_path: str) -> Dict[str, Set[str]]:
        splits_map = {"Train": "train", "Dev": "validation", "Test": "test"}
        signer_splits: Dict[str, Set[str]] = {
            "train": set(),
            "validation": set(),
            "test": set(),
        }
        if not os.path.exists(signer_meta_file_path):
            raise FileNotFoundError(
                f"Signer meta file not found: {signer_meta_file_path}"
            )
        with open(signer_meta_file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                person_id_str, split_key = row.get("ID"), row.get("Set")
                if person_id_str is None or split_key is None:
                    logger.warning(
                        f"Row in signer_meta.csv missing 'ID' or 'Set': {row}. Skipping."
                    )
                    continue
                if split_key in splits_map:
                    signer_splits[splits_map[split_key]].add(person_id_str)
                else:
                    logger.warning(
                        f"Unknown split value '{split_key}' for person ID '{person_id_str}'. Skipping."
                    )
        return signer_splits

    def _split_generators(self, dl_manager: Any) -> List[datasets.SplitGenerator]:
        if not hasattr(self.config, "data_dir") or not self.config.data_dir:
            raise ValueError("`data_dir` not passed to builder.")
        dataset_root = Path(self.config.data_dir)
        resolved_signer_meta_path = Path(self.SIGNER_META_PATH)
        if not resolved_signer_meta_path.exists():
            raise FileNotFoundError(
                f"Signer meta file not found: {resolved_signer_meta_path}, please review SIGNER_META_PATH variable in loaders/combined.py."
            )
        signer_to_split_map = self._load_signer_splits(str(resolved_signer_meta_path))
        split_generators = []
        split_definitions = {
            "train": datasets.Split.TRAIN,
            "validation": datasets.Split.VALIDATION,
            "test": datasets.Split.TEST,
        }
        for split_name_str, hf_split_enum in split_definitions.items():
            person_ids_for_this_split = signer_to_split_map.get(split_name_str, set())
            if not person_ids_for_this_split:
                logger.info(
                    f"No persons for split '{split_name_str}'. Split will be empty."
                )
            else:
                logger.info(
                    f"Found {len(person_ids_for_this_split)} persons for split '{split_name_str}'."
                )
            split_generators.append(
                datasets.SplitGenerator(
                    name=hf_split_enum,
                    gen_kwargs={
                        "dataset_root": dataset_root,
                        "person_ids_for_split": person_ids_for_this_split,
                        "current_split_name_str": split_name_str,
                    },
                )
            )
        if not split_generators:
            raise RuntimeError("No SplitGenerators created.")
        return split_generators

    def _generate_examples(
        self,
        dataset_root: Path,
        person_ids_for_split: Set[str],
        current_split_name_str: str,
    ) -> Iterator[Tuple[str, Dict]]:

        global_idx_this_split = 0

        if not person_ids_for_split:
            logger.info(
                f"No persons assigned to split '{current_split_name_str}'. Skipping example generation."
            )
            return

        for dataset_name, parser_info in DATASET_PARSERS_MAP.items():
            parse_function = parser_info["func"]
            person_prefix = parser_info["person_prefix"]

            sub_dataset_path = dataset_root / dataset_name

            if not sub_dataset_path.is_dir():
                logger.warning(
                    f"Sub-dataset folder '{sub_dataset_path}' not found. Skipping dataset '{dataset_name}'."
                )
                continue

            for item_in_sub_dataset in sub_dataset_path.iterdir():

                if (
                    not item_in_sub_dataset.is_dir()
                    or not item_in_sub_dataset.name.startswith(person_prefix)
                ):
                    continue

                person_folder_name = item_in_sub_dataset.name

                try:
                    person_id_num_str = person_folder_name[len(person_prefix) :]
                    if not person_id_num_str.isdigit():
                        continue
                except IndexError:
                    continue

                # dilter if person is not from actual split
                if person_id_num_str not in person_ids_for_split:
                    continue

                logger.debug(
                    f"Processing data for person '{person_folder_name}' from dataset '{dataset_name}' for split '{current_split_name_str}'"
                )

                try:
                    for annotation_obj in parse_function(
                        dataset_root, dataset_name, person_folder_name
                    ):
                        segment_id = f"{dataset_name}_{person_folder_name}_{current_split_name_str}_{global_idx_this_split}"

                        try:
                            final_video_path = Path(annotation_obj.video_path).resolve()
                        except Exception as path_e:
                            logger.error(
                                f"Error resolving video path '{annotation_obj.video_path}' for segment {segment_id}: {path_e}"
                            )
                            continue

                        yield global_idx_this_split, {
                            "segment_id": segment_id,
                            "video_path": str(final_video_path),
                            "start_time_ms": annotation_obj.start_time_ms,
                            "end_time_ms": annotation_obj.end_time_ms,
                            "label": annotation_obj.label,
                            "dataset_source": dataset_name,
                            "person_id": person_folder_name,
                        }
                        global_idx_this_split += 1
                except Exception as e:
                    logger.error(
                        f"Error processing annotations for {person_folder_name} in {dataset_name} (split {current_split_name_str}): {e}",
                        exc_info=True,
                    )
