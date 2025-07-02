import os
import tempfile
from pathlib import Path

import streamlit as st
from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord


class DataLoader:
    def __init__(self, saver_class: type):
        self.saver_class = saver_class
        self.extension = self.saver_class.get_file_extension().lstrip(".")

    def load(self, uploaded_file) -> AnnotationRecord | None:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{self.extension}"
            ) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            return self.saver_class.load_record(Path(tmp_path))
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
