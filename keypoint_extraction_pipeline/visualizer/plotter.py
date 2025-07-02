import numpy as np
import plotly.graph_objects as go
from keypoint_extraction_pipeline.schemas.keypoints import FrameLandmarks, LandmarkSet
from typing import Optional


class Plotter:
    def __init__(self, colors, marker_size):
        self.colors = colors
        self.marker_size = marker_size
        self.swap_axes = True

    def plot_frame(
        self, frame_landmarks_data: FrameLandmarks, selected_sets: list[str]
    ) -> tuple[go.Figure, bool]:
        fig = go.Figure()
        all_pts = []
        added = False

        for i, name in enumerate(selected_sets):
            landmark_set_to_plot: Optional[LandmarkSet] = getattr(
                frame_landmarks_data, name, None
            )

            if not landmark_set_to_plot or not landmark_set_to_plot.keypoints:
                continue
            coords = np.array([[p.x, p.y, p.z] for p in landmark_set_to_plot.keypoints])

            if self.swap_axes:
                # coords[:,0] = X, coords[:,1] = Y, coords[:,2] = Z
                coords_display = coords.copy()
                coords_display[:, 1] *= -1
                coords_display = coords_display[
                    :, [0, 2, 1]
                ]  # Swap Y and Z for visualization (X, Z, -Y)
            else:
                coords_display = coords

            x, y, z = coords_display[:, 0], coords_display[:, 1], coords_display[:, 2]
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(
                        size=self.marker_size, color=self.colors[i % len(self.colors)]
                    ),
                    name=name,
                )
            )
            all_pts.append(coords_display)
            added = True

        if added:
            pts = np.concatenate(all_pts, axis=0)
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)
            mid = (mins + maxs) / 2
            max_range = (maxs - mins).max() / 2
            ranges = [[mid[j] - max_range, mid[j] + max_range] for j in range(3)]
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=40),
                scene=dict(
                    xaxis=dict(range=ranges[0], title="X"),
                    yaxis=dict(range=ranges[1], title="Y"),
                    zaxis=dict(range=ranges[2], title="Z"),
                    aspectmode="cube",
                ),
                legend_title_text="Subsets",
            )

        return fig, added
