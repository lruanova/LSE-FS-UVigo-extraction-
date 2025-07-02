import logging
import csv

from typing import Optional

from .parser import AnnotationParser
from keypoint_extraction_pipeline.schemas.annotation import Annotation
import os


class CSVParser(AnnotationParser):
    """
    Parses annotations from a CSV file.
    """

    def __init__(
        self,
        video_id_field: str = "path",
        start_time_field: str = "start_time",
        end_time_field: str = "end_time",
        label_field: str = "label",
        videos_prefix: Optional[str] = None,
        process_full_video: bool = False,
    ):
        """
        Parses a CSV file where each row represents an annotation for a video segment.
        It expects identifiers for columns: [identifier, start_time, end_time, label].

        Attributes:
            video_id_field : The name of the CSV column containing the video identifier or path.
            start_time_field : The name of the CSV column containing the start time of the annotation (in milliseconds).
            end_time_field : The name of the CSV column containing the end time of the annotation (in milliseconds).
            label_field : The name of the CSV column containing the label for the annotation.
            videos_prefix : An optional prefix to prepend to relative video paths found in the CSV.
                                        If a video path in the CSV is absolute, this prefix is ignored for that path.
            process_full_video : If True, ignores start and end times and processes the entire video.
                                    Start and end times in the `Annotation` object will be `None`.
        """
        super().__init__(process_full_video)
        # logging.info(
        #     f"\
        #         ***CSV Parser Summary*** \n \
        #         \t process_full_video: {process_full_video} \n \
        #         \t videos_dir: {videos_prefix} \n \
        #     "
        # )
        self.video_id_field = video_id_field
        self.start_time_field = start_time_field
        self.end_time_field = end_time_field
        self.label_field = label_field
        self.videos_prefix = videos_prefix

    def parse(self, source: str):
        try:
            with open(source, mode="r") as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    try:
                        annotation = Annotation(
                            video_path=self._get_video_path(row[self.video_id_field]),
                            start_time_ms=(
                                float(row[self.start_time_field])
                                if not self.process_full_video
                                else None
                            ),
                            end_time_ms=(
                                float(row[self.end_time_field])
                                if not self.process_full_video
                                else None
                            ),
                            label=self._format_label(row[self.label_field]),
                        )
                        yield annotation
                    except KeyError as e:
                        missing_field = e.args[0]
                        error_msg = f"Field '{missing_field}' not found in CSV file"
                        logging.error(error_msg)
                        continue  # skip this row and proceed to the next
                    except ValueError as e:
                        error_msg = f"Invalid data format in CSV file at row {csv_reader.line_num}: {e}"
                        logging.error(error_msg)
                        continue  # skip this row
        except FileNotFoundError:
            error_msg = f"File '{source}' not found"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
        except Exception as e:
            logging.error(f"An error occurred while parsing the CSV file: {e}")
            raise e

    def _get_video_path(self, video_id: str) -> str:
        if self.videos_prefix and not os.path.isabs(video_id):
            video_path = os.path.join(self.videos_prefix, video_id)
            return os.path.normpath(video_path)
        return os.path.normpath(video_id)

    def _format_label(self, label: str) -> str:
        if label.startswith("DT:"):
            return label[3:].upper()
        return label.upper()


#
