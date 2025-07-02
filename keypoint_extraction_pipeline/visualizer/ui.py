import streamlit as st
from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord
from keypoint_extraction_pipeline.schemas.keypoints import FrameLandmarks


class KeypointUI:
    def __init__(self, viz_cfg):
        self.colors = list(viz_cfg.colors)
        self.marker_size = viz_cfg.get("marker_size", 4)

    def view_options(self):
        swap = st.checkbox(
            "🔄 Do not swap Y/Z",
            value=True,
            help="Change Y and Z axis to visualizar",
        )
        return swap

    def file_uploader(self, extension: str):
        uploaded = st.file_uploader(
            f"Choose .{extension} files", type=[extension], accept_multiple_files=True
        )
        return uploaded or []

    def data_card(self, file_name: str, record: AnnotationRecord):
        st.markdown(f"**📄 {file_name}**")
        cols = st.columns(4)
        md = record.metadata

        # Segment ID
        cols[0].markdown(f"**ID:** `{md.segment_id}`")

        # Label
        cols[1].markdown(f"**Label:** {md.label}")

        # Handedness
        cols[2].markdown(f"**Handedness:** {md.handness or 'N/A'}")

        # Duration (seconds)
        if md.start_time_ms is not None and md.end_time_ms is not None:
            duration_s = (md.end_time_ms - md.start_time_ms) / 1000
            cols[3].markdown(f"**Duration:** {duration_s:.2f} s")
        else:
            cols[3].markdown("**Duration:** N/A")

    def frame_and_subset_selector(self, record: AnnotationRecord):
        frames = record.frames
        frame_idx = (
            st.slider("Frame", 0, len(frames) - 1, 0, key="frame_slider")
            if len(frames) > 1
            else 0
        )
        current_frame_landmarks: FrameLandmarks = frames[frame_idx]
        # Get available landmark sets by checking which fields in FrameLandmarks are not None
        available = [
            field_name
            for field_name in FrameLandmarks.model_fields.keys()
            if getattr(current_frame_landmarks, field_name) is not None
        ]
        default = [
            s for s in available if "hand" in s and "velocity" not in s
        ] or available
        subsets = st.multiselect(
            "Keypoint sets",
            options=available,
            default=default,
            key="subset_multiselect",
        )
        return frame_idx, subsets

    def display_static(self, fig):
        st.subheader("Static 3D View")
        st.plotly_chart(fig, use_container_width=True)

    def display_animation(self, fig):
        st.subheader("Animated 3D View")
        st.plotly_chart(fig, use_container_width=True)

    def display_record_json_toggle(self, record: AnnotationRecord):
        if st.checkbox(
            "Show full AnnotationRecord (metadata + FrameLandmarks)",
        ):
            st.json(record.model_dump_json(indent=2))
