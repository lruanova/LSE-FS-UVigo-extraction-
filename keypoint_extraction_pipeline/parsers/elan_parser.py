import logging
import pympi.Elan as elan
from typing import Iterator

from .parser import AnnotationParser
from keypoint_extraction_pipeline.schemas.annotation import Annotation


class ELANParser(AnnotationParser):
    def __init__(
        self,
        tier_name: str = "M_Glosa",
        filter_dt=True,
        read_source_from_eaf: bool = False,
        process_full_video=False,
    ):
        super().__init__(process_full_video)
        self.tier_name = tier_name
        self.filter_dt = filter_dt
        self.read_source_from_eaf = read_source_from_eaf

    def parse(self, source: str) -> Iterator[Annotation]:
        eaf_reader = elan.Eaf(source.replace("mp4", "eaf"))
        try:
            eaf_annotations = eaf_reader.get_annotation_data_for_tier(self.tier_name)
            for annotation in eaf_annotations:
                if self.filter_dt and not annotation[2].startswith("DT:"):
                    continue
                yield Annotation(
                    video_path=(
                        eaf_reader.get_linked_files()[0]["MEDIA_URL"]
                        if self.read_source_from_eaf
                        else source
                    ),
                    start_time_ms=annotation[0],
                    end_time_ms=annotation[1],
                    label=annotation[2],
                )
        except KeyError:
            error_msg = f"Tier '{self.tier_name}' not found in ELAN file"
            logging.error(error_msg)
            raise ValueError(error_msg)
