from typing import Iterable, Optional
from abc import abstractmethod

class FrameExtractor():
    """Defines the interface for a frame extractor."""
    @abstractmethod
    def extract_segment(
        self, 
        video_path: str, 
        start_time_ms: Optional[float] = None, 
        end_time_ms: Optional[float] = None,
    ) -> Iterable:
        """
        Returns a iterable of a segment of a video.
        
        Args:
            video_path: the path of the input video from where to extract the segment.
            start_time_ms: start of the segment (in milliseconds)
            end_time_ms: end of the segment (in milliseconds)
        """
        pass

    @abstractmethod
    def extract_image(self, img_path : str ) -> Iterable:
        """
        Returns a single frame as an interable.

        Args:
            img_path: path of the image to read.

        """
        pass