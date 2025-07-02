import streamlit as st
from omegaconf import DictConfig
import hydra
from hydra.utils import get_class
from keypoint_extraction_pipeline.visualizer.data_loader import DataLoader
from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord
from keypoint_extraction_pipeline.visualizer.ui import KeypointUI
from keypoint_extraction_pipeline.visualizer.animator import Animator


class KeypointVisualizerApp:
    def __init__(self, cfg: DictConfig):
        st.set_page_config(layout=cfg.visualizer.layout)
        self.cfg = cfg
        saver_cls = get_class(cfg.saver._target_)
        self.loader = DataLoader(saver_cls)
        self.ui = KeypointUI(cfg.visualizer)
        self.animator = Animator(cfg.visualizer)

    def run(self):
        st.title("Keypoint Visualizer")
        files = self.ui.file_uploader(self.loader.extension)
        for f in files:
            record: AnnotationRecord | None = self.loader.load(f)
            if not record:
                continue

            # Summary card
            self.ui.data_card(f.name, record)

            # Frame selector
            if not record.frames:
                st.warning("No frames found in this record.")
                continue
            frame_idx, subsets = self.ui.frame_and_subset_selector(record)

            # change view
            swap_axes = self.ui.view_options()

            col1, col2 = st.columns(2)
            # Static keypoint view
            with col1:
                static_fig = self.animator.plot_static(
                    record.frames[frame_idx], subsets, swap_axes=swap_axes
                )
                if static_fig:
                    self.ui.display_static(static_fig)
            with col2:
                # Animtation
                anim_fig = self.animator.plot_animation(
                    record, subsets, swap_axes=swap_axes
                )
                if anim_fig:
                    self.ui.display_animation(anim_fig)

            st.write("#")  # separator
            # JSON summary
            self.ui.display_record_json_toggle(record=record)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    KeypointVisualizerApp(cfg).run()


if __name__ == "__main__":
    main()
