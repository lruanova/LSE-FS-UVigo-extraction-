import plotly.graph_objects as go
import numpy as np
from keypoint_extraction_pipeline.schemas.annotation import AnnotationRecord
from keypoint_extraction_pipeline.schemas.keypoints import FrameLandmarks
from keypoint_extraction_pipeline.visualizer.plotter import Plotter


class Animator:
    def __init__(self, viz_cfg):
        self.plotter = Plotter(viz_cfg.colors, viz_cfg.get("marker_size", 4))

    def plot_static(
        self,
        frame_landmarks_data: FrameLandmarks,
        subsets: list[str],
        swap_axes: bool = True,
    ):
        self.plotter.swap_axes = swap_axes
        fig, ok = self.plotter.plot_frame(frame_landmarks_data, subsets)
        return fig if ok else None

    def plot_animation(
        self, record: AnnotationRecord, subsets: list[str], swap_axes: bool = True
    ) -> go.Figure | None:
        self.plotter.swap_axes = swap_axes

        frames = []
        all_pts = []

        for idx, frame_landmarks_data_item in enumerate(record.frames):
            fig_f, ok = self.plotter.plot_frame(frame_landmarks_data_item, subsets)
            if not ok:
                continue
            frames.append(go.Frame(data=fig_f.data, name=str(idx)))
            for tr in fig_f.data:
                arr = np.vstack([tr.x, tr.y, tr.z]).T  # type: ignore
                all_pts.append(arr)

        if not frames:
            return None

        all_pts = np.concatenate(all_pts, axis=0)
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        mid = (mins + maxs) / 2
        max_range = (maxs - mins).max() / 2
        ranges = [[mid[j] - max_range, mid[j] + max_range] for j in range(3)]

        fig = go.Figure(data=frames[0].data, frames=frames)
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=50, redraw=True),
                                    fromcurrent=True,
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(frame=dict(duration=0), mode="immediate"),
                            ],
                        ),
                    ],
                )
            ],
            scene=dict(
                xaxis=dict(range=ranges[0], title="X"),
                yaxis=dict(range=ranges[1], title="Y"),
                zaxis=dict(range=ranges[2], title="Z"),
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        return fig
