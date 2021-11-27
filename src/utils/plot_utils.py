# Adapted from starter project

import torch 
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots

def plot_hand(img, K, kp3d):
    kp2d = kp3d_to_kp2d(kp3d, K)
    return plot_fingers(kp2d, img_rgb=img)

def plot_hand_3D(preds, target=None):
        
    fig = make_subplots()
    i = 0

    for points in [preds, target] if target is not None else [preds]:
        
        root_point = go.Scatter3d(
            x=[points[0,0]], y=[points[0,1]], z=[points[0,2]],
            mode='markers',
            marker=dict(size=5, color='#2d3436' if i == 0 else '#b2bec3')
        )
        
        palm = [0,1,2,3,4,5]
        palm_line = go.Scatter3d(
            x=torch.cat((points[palm,0], points[0,0].view(1))), 
            y=torch.cat((points[palm,1], points[0,1].view(1))), 
            z=torch.cat((points[palm,2], points[0,2].view(1))), 
            mode='lines', 
            line_width=2, 
            line_color='#636e72' if i == 0 else '#b2bec3'
        )
        
        thumb = [1,6,11,16]
        thumb_line = go.Scatter3d(
            x=points[thumb,0], y=points[thumb,1], z=points[thumb,2], mode='lines', 
            line_width=2, 
            line_color='#d63031' if i == 0 else '#b2bec3'
        )
        thumb_point = go.Scatter3d(
            x=points[thumb,0], y=points[thumb,1], z=points[thumb,2],
            mode='markers',
            marker=dict(size=5, color='#ff7675' if i == 0 else '#b2bec3')
        )
        
        index = [2,7,12,17]  
        index_line = go.Scatter3d(
            x=points[index,0], y=points[index,1], z=points[index,2], mode='lines', 
            line_width=2, 
            line_color='#00cec9' if i == 0 else '#b2bec3'
        )
        index_point = go.Scatter3d(
            x=points[index,0], y=points[index,1], z=points[index,2],
            mode='markers',
            marker=dict(size=5, color='#81ecec' if i == 0 else '#b2bec3')
        )   

        middle = [3,8,13,18]
        middle_line = go.Scatter3d(
            x=points[middle,0], y=points[middle,1], z=points[middle,2], mode='lines', 
            line_width=2, 
            line_color='#0984e3' if i == 0 else '#b2bec3'
        )
        middle_point = go.Scatter3d(
            x=points[middle,0], y=points[middle,1], z=points[middle,2],
            mode='markers',
            marker=dict(size=5, color='#74b9ff' if i == 0 else '#b2bec3') 
        )
        
        ring = [4,9,14,19]
        ring_line = go.Scatter3d(
            x=points[ring,0], y=points[ring,1], z=points[ring,2], mode='lines', 
            line_width=2, 
            line_color='#6c5ce7' if i == 0 else '#b2bec3'
        )
        ring_point = go.Scatter3d(
            x=points[ring,0], y=points[ring,1], z=points[ring,2],
            mode='markers',
            marker=dict(size=5, color='#a29bfe' if i == 0 else '#b2bec3')
        )
        
        pinky = [5, 10, 15, 20]
        pinky_line = go.Scatter3d(
            x=points[pinky,0], y=points[pinky,1], z=points[pinky,2], mode='lines', 
            line_width=2, 
            line_color='#e17055' if i == 0 else '#b2bec3'
        )
        pinky_point = go.Scatter3d(
            x=points[pinky,0], y=points[pinky,1], z=points[pinky,2],
            mode='markers',
            marker=dict(size=5, color='#fab1a0' if i == 0 else '#b2bec3')
        )
        
        fig.add_trace(root_point)
        fig.add_trace(palm_line)
        fig.add_trace(thumb_line)
        fig.add_trace(thumb_point)
        fig.add_trace(index_line)
        fig.add_trace(index_point)
        fig.add_trace(middle_line)
        fig.add_trace(middle_point)
        fig.add_trace(ring_line)
        fig.add_trace(ring_point)
        fig.add_trace(pinky_line)
        fig.add_trace(pinky_point)
        i+=1


    fig.update_layout(width=600, height=600, showlegend=False)
    return fig 

def kp3d_to_kp2d(kp3d, K):
    """
    Pinhole camera model projection
    K: camera intrinsics (3 x 3)
    kp3d: 3D coordinates wrt to camera (n_kp x 3)
    """
    kp2d = (kp3d @ K.T) / kp3d[..., 2:3]

    return kp2d[..., :2]

def plot_fingers(kp, **kwargs):
    assert len(kp.shape) == 2

    if kp.shape[1] == 2:
        # 2D keypoints
        return plot_fingers2D(kp, **kwargs)
    elif kp.shape[1] == 3:
        # 3D keypoints
        return plot_fingers3D(kp, **kwargs)
    else:
        raise Exception(f"Invalid keypoint dimensionality: {kp.shape[1]}")


def plot_fingers2D(kp2d, img_rgb=None, ax=None, c="gt"):
    """
    Plots the 2D keypoints over the image. 
    """

    assert len(kp2d.shape) == 2, "plot_fingers2D does not accept batch input"

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    if not img_rgb is None:
        ax.clear()
        ax.imshow(img_rgb)

    if c == "pred":
        c = ["#660000", "#b30000", "#ff0000", "#ff4d4d", "#ff9999"]
    elif c == "gt":
        c = ["#000066", "#0000b3", "#0000ff", "#4d4dff", "#9999ff"]
    else:
        assert isinstance(c, list)

    for i in range(5):
        idx_to_plot = np.arange(i + 1, 21, 5)
        to_plot = np.concatenate((kp2d[0:1], kp2d[idx_to_plot]), axis=0)
        ax.plot(to_plot[:, 0], to_plot[:, 1], "x-", color=c[i])

    return fig


def plot_fingers3D(kp3d, ax=None, c="gt", lims=None, clear=True, view=(-90, 0)):
    """
    Plots the 3D keypoints over the image.
    """

    assert len(kp3d.shape) == 2, "plot_fingers does not accept batch input"

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    if c == "pred":
        c = ["#660000", "#b30000", "#ff0000", "#ff4d4d", "#ff9999"]
    elif c == "gt":
        c = ["#000066", "#0000b3", "#0000ff", "#4d4dff", "#9999ff"]
    else:
        assert isinstance(c, list)

    min_range = -3
    max_range = 3
    for i in range(5):
        idx_to_plot = np.arange(i + 1, 21, 5)
        to_plot = np.concatenate((kp3d[0:1], kp3d[idx_to_plot]), axis=0)
        ax.plot(to_plot[:, 0], to_plot[:, 1], to_plot[:, 2], "x-", color=c[i])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if not lims is None:
        min_range, max_range = lims
        ax.set_xlim(min_range, max_range)
        ax.set_ylim(min_range, max_range)
        ax.set_zlim(min_range, max_range)

    if not view is None:
        """
        view=(-90,0): View from above to see depth error more clearly
        view=(-90,-90): View from front to see camera view. 
        It is very similar to plotting 2D keypoints, hence less informative if plotting
        2D keypoints already.
        """
        azim, elev = view
        ax.view_init(azim=azim, elev=elev)

    set_axes_equal(ax)

    return fig


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().

    Source: https://stackoverflow.com/a/31364297/1550099
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
