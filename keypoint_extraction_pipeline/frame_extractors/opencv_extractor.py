import logging
import os
from pathlib import Path
from typing import Iterable, Optional
import numpy as np
from omegaconf import DictConfig
import cv2
from keypoint_extraction_pipeline.frame_extractors.extractor import FrameExtractor

logging.getLogger().setLevel(logging.INFO)


class OpenCVFrameExtractor(FrameExtractor):
    def __init__(self, reader_config: DictConfig):
        super().__init__()
        self.cfg = reader_config
        logging.debug(f"OpenCVFrameExtractor initialized with config: {self.cfg}")

    def _ms_to_frames(self, ms: float, fps: float) -> int:
        if fps == 0:
            return 0
        return int(ms / 1000 * fps)

    def extract_segment(
        self,
        video_path: str,
        start_time_ms: Optional[float] = None,
        end_time_ms: Optional[float] = None,
    ) -> Iterable[np.ndarray]:
        """
        Returns a iterable of a segment of a video. Reads the video using
        OpenCV and gets the fps. Gets the segment start-end frames from
        input timestamps and returns each as an iterable.

        Args:
            video_path: the path of the input video from where to extract the segment.
            start_time_ms: start of the segment (in milliseconds)
            end_time_ms: end of the segment (in milliseconds)

        """
        if os.path.isdir(video_path):
            yield from self.extract_image(img_dir=video_path)
            return  # stop here to avoid running following code

        cap = None
        total_frames_video_for_print = 0
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logging.error(f"OpenCV: Cannot open video {video_path}.")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames_video_for_print = total_frames_video

            if fps <= 0:
                logging.error(
                    f"OpenCV: Invalid FPS ({fps}) for video {video_path}. Cannot extract segment by time."
                )
                if cap:
                    cap.release()
                return

            if start_time_ms is not None and not np.isnan(start_time_ms):
                start_frame = self._ms_to_frames(start_time_ms, fps=fps)
            else:
                start_frame = 0

            if end_time_ms is not None and not np.isnan(end_time_ms):
                end_frame = self._ms_to_frames(end_time_ms, fps=fps)
            else:
                end_frame = total_frames_video - 1 if total_frames_video > 0 else -1

            start_frame = max(0, start_frame)
            if total_frames_video > 0:
                end_frame = min(total_frames_video - 1, end_frame)
            else:
                end_frame = -1

            logging.info(
                f"OpenCV Processing: Video '{Path(video_path).name}', FPS={fps:.2f}, TotalFrames={total_frames_video_for_print}, "
                f"ReqStartMs={start_time_ms}, ReqEndMs={end_time_ms} -> "
                f"CalcStartFrame={start_frame}, CalcEndFrame={end_frame}"
            )

            if start_frame > end_frame:
                logging.warning(
                    f"OpenCV: Video '{Path(video_path).name}': Start frame {start_frame} is after end frame {end_frame}. Yielding NO frames."
                )
                return

            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            current_frame_read_idx = start_frame
            frames_yielded_count = 0
            while current_frame_read_idx <= end_frame:
                readed_ok, frame = cap.read()
                if not readed_ok:
                    logging.warning(
                        f"OpenCV: Video '{Path(video_path).name}': cap.read() failed at frame index {current_frame_read_idx} "
                        f"(target end: {end_frame}). Yielded {frames_yielded_count} frames."
                    )
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_rgb
                frames_yielded_count += 1
                current_frame_read_idx += 1

            logging.debug(
                f"OpenCV: Finished segment for '{Path(video_path).name}'. Yielded {frames_yielded_count} frames from range [{start_frame}-{end_frame}]."
            )

        except Exception as e:
            logging.error(
                f"OpenCV: Error during extract_segment for '{Path(video_path).name}': {e}",
                exc_info=True,
            )
        finally:
            if cap:
                cap.release()

    def extract_image(self, img_dir: str):
        """
        Returns a single frame as an interable. Reads the image
        path using opencv and returns it as a rgb frame.

        Args:
            img_path: path of the image to read.

        """
        img_files = sorted(
            p
            for p in Path(img_dir).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not img_files:
            logging.warning(f"[DirFrames] sin imágenes en {img_dir}")
            return

        logging.info(f"[DirFrames] {len(img_files)} imgs en {Path(img_dir).name}")
        for img_file in img_files:
            img_bgr = cv2.imread(str(img_file))
            if img_bgr is None:
                logging.warning(f"No pude leer {img_file}")
                continue
            yield cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return
