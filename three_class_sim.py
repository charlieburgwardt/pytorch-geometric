# %%
"""
Three-class GMTI simulation generator

Classes:
  - background (label=0): randomly positioned, random heading relative to x-axis
  - column (label=1): group of targets aligned along a baseline angle; heading equals baseline angle
  - line abreast (label=2): group of targets aligned along a baseline angle; heading equals baseline angle + 90 degrees

Notes:
  * Baseline angle defines the spatial alignment of the group (the line the points sit on).
  * Heading defines the direction of travel used to compute LOS velocity.
  * Groups are formed as in the original code: POINTS_PER_GROUP targets spaced along a line.
"""

import math
import numpy as np
import pandas as pd

# -------------------------------------------------------------
# LOS velocity utility (supports scalar or vector inputs)
# -------------------------------------------------------------
def calculate_los_velocity(x_t, y_t, theta_t, v_t, x_s, y_s, z_s):
    """
    Calculate Line-of-Sight (LOS) velocity based on STANAG 4607 single dwell geometry.

    Parameters
    ----------
    x_t, y_t : array-like or float
        Target position coordinates.
    theta_t : array-like or float
        Target heading in radians (angle from x-axis).
    v_t : array-like or float
        Target speed (m/s).
    x_s, y_s, z_s : float
        Sensor position coordinates (z_s is altitude).

    Returns
    -------
    array-like or float
        LOS velocity.
    """
    dx = np.asarray(x_t) - x_s
    dy = np.asarray(y_t) - y_s
    los_distance = np.sqrt(dx**2 + dy**2 + z_s**2)
    return np.asarray(v_t) * ((dx * np.cos(theta_t)) + (dy * np.sin(theta_t))) / los_distance

# -------------------------------------------------------------
# Simulation parameters
# -------------------------------------------------------------
PLANE_SIZE = 5000.0  # meters (square plane: 0..5000 in both x and y)
N_BACKGROUND = 100
N_GROUPS = 10
POINTS_PER_GROUP = 10
POINT_SPACING = 10.0  # meters
V_TARGET_MIN, V_TARGET_MAX = 5.0, 25.0  # m/s

# Fixed sensor (single dwell)
x_sensor = 0.0
y_sensor = 0.0
z_sensor = 1000.0

# Reproducibility (uncomment to fix seed)
# rng = np.random.default_rng(20251112)
rng = np.random.default_rng()

# Desired column order (kept consistent with original)
COLS = [
    'x_sensor', 'y_sensor', 'z_sensor',
    'x_target', 'y_target', 'z_target',
    'LOS_Velocity', 'label', 'dwell_id', 'heading_deg', 'velocity'
]


def simulate_three_classes(num_dwells: int = 3) -> pd.DataFrame:
    df_all = pd.DataFrame(columns=COLS)

    # Precompute centered offsets for groups
    t_values = (np.arange(POINTS_PER_GROUP) - (POINTS_PER_GROUP - 1)/2.0) * POINT_SPACING
    half_extent = (POINTS_PER_GROUP - 1) * POINT_SPACING / 2.0  # 45 m for 10 points at 10 m spacing

    for dwell_id in range(num_dwells):
        # ---------------------------
        # Background points (label = 0)
        # ---------------------------
        background_x = rng.uniform(0.0, PLANE_SIZE, N_BACKGROUND)
        background_y = rng.uniform(0.0, PLANE_SIZE, N_BACKGROUND)
        background_v = rng.uniform(V_TARGET_MIN, V_TARGET_MAX, N_BACKGROUND)
        background_heading = rng.uniform(0.0, 359.9, size=N_BACKGROUND)
        background_hdg_rad = np.deg2rad(background_heading)
        background_los_vel = calculate_los_velocity(
            background_x, background_y, background_hdg_rad, background_v,
            x_sensor, y_sensor, z_sensor
        )

        # ---------------------------
        # Column groups (label = 1)
        # ---------------------------
        col_x_all, col_y_all, col_los_all = [], [], []
        col_label_all, col_heading_all, col_vel_all = [], [], []

        for _ in range(N_GROUPS):
            baseline_deg = rng.uniform(0.0, 359.9)
            baseline_rad = np.deg2rad(baseline_deg)
            ux, uy = np.cos(baseline_rad), np.sin(baseline_rad)  # spatial alignment

            # Group center sampled with a fixed margin so all points stay within [0, PLANE_SIZE]
            margin = half_extent
            cx = rng.uniform(margin, PLANE_SIZE - margin)
            cy = rng.uniform(margin, PLANE_SIZE - margin)

            # Points for this group aligned along the baseline
            x_group = cx + t_values * ux
            y_group = cy + t_values * uy

            # Single target velocity for the group (same speed per point)
            v_group_scalar = rng.uniform(V_TARGET_MIN, V_TARGET_MAX)
            v_group = np.full(POINTS_PER_GROUP, v_group_scalar)

            # Heading equals baseline angle
            heading_rad = np.full(POINTS_PER_GROUP, baseline_rad)
            los_vals = calculate_los_velocity(
                x_group, y_group, heading_rad, v_group,
                x_sensor, y_sensor, z_sensor
            )

            # Collect
            col_x_all.append(x_group)
            col_y_all.append(y_group)
            col_los_all.append(los_vals)
            col_label_all.append(np.ones(POINTS_PER_GROUP, dtype=int))
            col_heading_all.append(np.full(POINTS_PER_GROUP, baseline_deg))
            col_vel_all.append(v_group)

        # Concatenate column groups
        col_x_all = np.concatenate(col_x_all)
        col_y_all = np.concatenate(col_y_all)
        col_los_all = np.concatenate(col_los_all)
        col_label_all = np.concatenate(col_label_all)
        col_heading_all = np.concatenate(col_heading_all)
        col_vel_all = np.concatenate(col_vel_all)

        # ---------------------------
        # Line abreast groups (label = 2)
        # ---------------------------
        la_x_all, la_y_all, la_los_all = [], [], []
        la_label_all, la_heading_all, la_vel_all = [], [], []

        for _ in range(N_GROUPS):
            baseline_deg = rng.uniform(0.0, 359.9)
            baseline_rad = np.deg2rad(baseline_deg)
            ux, uy = np.cos(baseline_rad), np.sin(baseline_rad)  # spatial alignment

            # Group center with safe margin
            margin = half_extent
            cx = rng.uniform(margin, PLANE_SIZE - margin)
            cy = rng.uniform(margin, PLANE_SIZE - margin)

            # Points aligned along the baseline; travel perpendicular to baseline
            x_group = cx + t_values * ux
            y_group = cy + t_values * uy

            # Single target velocity for the group
            v_group_scalar = rng.uniform(V_TARGET_MIN, V_TARGET_MAX)
            v_group = np.full(POINTS_PER_GROUP, v_group_scalar)

            # Heading = baseline + 90 degrees (unique to line abreast motion signature)
            #heading_rad = np.full(POINTS_PER_GROUP, baseline_rad + math.radians(90))
            # Heading = baseline + 90 degrees, wrapped to [0, 359.9]
            heading_deg_wrapped = (baseline_deg + 90.0) % 360.0
            if heading_deg_wrapped >= 360.0:
                heading_deg_wrapped = 359.9
            heading_rad = np.full(POINTS_PER_GROUP, math.radians(heading_deg_wrapped))

            los_vals = calculate_los_velocity(
                x_group, y_group, heading_rad, v_group,
                x_sensor, y_sensor, z_sensor
            )

            # Collect
            la_x_all.append(x_group)
            la_y_all.append(y_group)
            la_los_all.append(los_vals)
            la_label_all.append(np.full(POINTS_PER_GROUP, 2, dtype=int))
            la_heading_all.append(np.full(POINTS_PER_GROUP, heading_deg_wrapped))
            la_vel_all.append(v_group)

        # Concatenate line abreast groups
        la_x_all = np.concatenate(la_x_all)
        la_y_all = np.concatenate(la_y_all)
        la_los_all = np.concatenate(la_los_all)
        la_label_all = np.concatenate(la_label_all)
        la_heading_all = np.concatenate(la_heading_all)
        la_vel_all = np.concatenate(la_vel_all)

        # ---------------------------
        # Assemble full dwell DataFrame
        # ---------------------------
        x_target_all = np.concatenate([background_x, col_x_all, la_x_all])
        y_target_all = np.concatenate([background_y, col_y_all, la_y_all])
        los_all = np.concatenate([background_los_vel, col_los_all, la_los_all])
        label_all = np.concatenate([
            np.zeros(N_BACKGROUND, dtype=int),
            col_label_all,
            la_label_all
        ])
        heading_all = np.concatenate([
            background_heading,
            col_heading_all,
            la_heading_all
        ])
        velocity_all = np.concatenate([
            background_v,
            col_vel_all,
            la_vel_all
        ])

        n_total = len(x_target_all)
        df = pd.DataFrame({
            'x_sensor': np.full(n_total, x_sensor),
            'y_sensor': np.full(n_total, y_sensor),
            'z_sensor': np.full(n_total, z_sensor),
            'x_target': x_target_all,
            'y_target': y_target_all,
            'z_target': np.zeros(n_total),
            'LOS_Velocity': los_all,
            'label': label_all,
            'dwell_id': np.full(n_total, dwell_id),
            'heading_deg': heading_all,
            'velocity': velocity_all,
        })[COLS]

        df_all = pd.concat([df_all, df], ignore_index=True)

    return df_all


if __name__ == "__main__":
    df_out = simulate_three_classes(num_dwells=100)
    # Remove cols as needed
    # For training remove these
    #df_out.drop(columns=['heading_deg', 'velocity'], inplace=True)
    #For inference without metrics drop label like this:
    df_out.drop(columns=['label', 'heading_deg', 'velocity'], inplace=True)

    # Save to a portable relative path; update to your Windows path if desired
    output_csv = "C:/Users/charlie.burgwardt/OneDrive - NV5/GMTI/Data/three_class_dwells.csv"
    df_out.to_csv(output_csv, index=False)
    print(f"Simulation complete. Data saved to {output_csv}")
