"""
gaussian_disaster_simulator.py
Gaussian Process (GP) Spatial Disaster Hazard Field & Correlated Edge Blockage Generator
based on the Gaussian Traveler Problem (GTP) framework with Posterior Conditioning (Kriging),
Anisotropic Kernels, Safe Origin/Destination Anchoring, and Structured Disaster Topologies
(Earthquake, Tsunami, Flood).

References:
- Dey, D., Shankar, K., & Bagnell, J. A. (2014). Gaussian Traveler Problem.
- Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull


# ==========================================
# 1. Isotropic & Anisotropic GP Kernels
# ==========================================

def rbf_kernel(X1, X2, length_scale=0.15, variance=1.5):
    """
    Computes the Squared Exponential (Radial Basis Function) covariance matrix K(X1, X2).
    Supports:
      - Isotropic length_scale: scalar l > 0
            k(x_i, x_j) = sigma_f^2 * exp( - ||x_i - x_j||^2 / (2 * l^2) )
      - Axis-aligned Anisotropic (ARD) length_scale: 2-tuple (l_x, l_y) with l_x != l_y
            k(x_i, x_j) = sigma_f^2 * exp( -0.5 * [ (dx / l_x)^2 + (dy / l_y)^2 ] )
      - Oriented Anisotropic length_scale: 3-tuple (l_parallel, l_perp, theta_rad)
            projects coordinates onto the coastline tangent/normal axes before scaling.

    Args:
        X1 (np.ndarray): Array of shape (N, 2) with 2D spatial coordinates.
        X2 (np.ndarray): Array of shape (M, 2) with 2D spatial coordinates.
        length_scale (float or tuple/list/np.ndarray): Scalar l, (l_x, l_y), or (l_para, l_perp, theta).
        variance (float): Signal variance sigma_f^2 > 0 controlling hazard amplitude.

    Returns:
        K (np.ndarray): Covariance matrix of shape (N, M).
    """
    X1 = np.atleast_2d(np.asarray(X1, dtype=float))
    X2 = np.atleast_2d(np.asarray(X2, dtype=float))

    ls = np.asarray(length_scale, dtype=float).ravel()
    if ls.size == 1:
        l_val = float(ls[0])
        sq_dists = cdist(X1, X2, metric='sqeuclidean')
        return variance * np.exp(-0.5 * sq_dists / (l_val ** 2))
    elif ls.size == 2:
        ls_vec = ls.reshape(1, 2)
        X1_scaled = X1 / ls_vec
        X2_scaled = X2 / ls_vec
        sq_dists_aniso = cdist(X1_scaled, X2_scaled, metric='sqeuclidean')
        return variance * np.exp(-0.5 * sq_dists_aniso)
    else:
        # Oriented anisotropic kernel: (l_parallel, l_perp, theta_rad)
        l_para, l_perp, theta = float(ls[0]), float(ls[1]), float(ls[2])
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])  # columns are u_parallel, u_perp
        ls_vec = np.array([[l_para, l_perp]])
        X1_rot = (X1 @ R) / ls_vec
        X2_rot = (X2 @ R) / ls_vec
        sq_dists_rot = cdist(X1_rot, X2_rot, metric='sqeuclidean')
        return variance * np.exp(-0.5 * sq_dists_rot)


def anisotropic_rbf_kernel(X1, X2, l_x=0.15, l_y=0.50, variance=1.5):
    """
    Explicit wrapper for the Anisotropic Squared Exponential (ARD) kernel:
        k(x_i, x_j) = sigma_f^2 * exp( -0.5 * ( (x_i - x_j)^2 / l_x^2 + (y_i - y_j)^2 / l_y^2 ) )
    """
    return rbf_kernel(X1, X2, length_scale=(l_x, l_y), variance=variance)


def sample_gp_prior(coordinates, length_scale=0.15, variance=1.5, jitter=1e-6, rng=None):
    """
    Samples an unconstrained latent disaster intensity vector Y ~ N(0, K) from a
    zero-mean Gaussian Process prior over the given 2D spatial coordinates (Dey et al., 2014).

    Args:
        coordinates (np.ndarray): Array of shape (N, 2) representing spatial query points X.
        length_scale (float or tuple): Isotropic l or anisotropic (l_x, l_y).
        variance (float): RBF kernel variance sigma_f^2.
        jitter (float): Small diagonal nugget added to K for positive-definite stability.
        rng (np.random.Generator or None): Optional NumPy random generator.

    Returns:
        Y (np.ndarray): Sampled latent hazard intensities of shape (N,).
        K (np.ndarray): Covariance matrix of shape (N, N).
    """
    coords = np.atleast_2d(np.asarray(coordinates, dtype=float))
    n_points = coords.shape[0]

    K = rbf_kernel(coords, coords, length_scale=length_scale, variance=variance)
    K_stable = K + jitter * np.eye(n_points)

    mean = np.zeros(n_points, dtype=float)
    if rng is None:
        Y = np.random.multivariate_normal(mean, K_stable)
    else:
        Y = rng.multivariate_normal(mean, K_stable)

    return Y, K


def compute_gp_posterior(
    X_u,
    X_L,
    t_L,
    length_scale=0.15,
    variance=1.5,
    noise_variance=1e-2,
    prior_mean=-3.0,
    jitter=1e-6
):
    """
    Computes the exact Gaussian Process posterior mean and covariance at unobserved
    query coordinates X_u conditioned on observed anchor coordinates X_L with target
    latent observations t_L (Dey et al., 2014; Rasmussen & Williams, 2006):

        y_bar_u = m_0 + k(X_u)^T (sigma^2 I + K_LL)^{-1} (t_L - m_0)
        Sigma_u = k(X_u, X_u) - k(X_u)^T (sigma^2 I + K_LL)^{-1} k(X_u)

    Using prior_mean m_0 = safe_y (-3.0) ensures that as distance from the disaster
    anchors increases (k(X_u) -> 0), the latent hazard field decays monotonically
    all the way down to the safe baseline m_0 = -3.0 rather than rebounding to 0.0.
    """
    X_u = np.atleast_2d(np.asarray(X_u, dtype=float))
    X_L = np.atleast_2d(np.asarray(X_L, dtype=float))
    t_L = np.asarray(t_L, dtype=float).ravel()

    n_L = X_L.shape[0]
    n_u = X_u.shape[0]

    # Prior covariance among anchors K_LL and regularized matrix (sigma^2 I + K_LL)
    K_LL = rbf_kernel(X_L, X_L, length_scale=length_scale, variance=variance)
    K_LL_reg = K_LL + (noise_variance + jitter) * np.eye(n_L)

    # Cross-covariance k(x_u)^T = K(X_u, X_L) of shape (U, L)
    K_uL = rbf_kernel(X_u, X_L, length_scale=length_scale, variance=variance)

    # Prior covariance among unobserved query locations k(x_u, x_u) = K(X_u, X_u)
    K_uu = rbf_kernel(X_u, X_u, length_scale=length_scale, variance=variance)

    # Solve (sigma^2 I + K_LL)^{-1} (t_L - prior_mean) and (sigma^2 I + K_LL)^{-1} k(X_u)
    alpha = np.linalg.solve(K_LL_reg, t_L - prior_mean)
    V = np.linalg.solve(K_LL_reg, K_uL.T)  # shape (L, U)

    # Posterior mean: y_bar_u = m_0 + k(X_u)^T (sigma^2 I + K_LL)^{-1} (t_L - m_0)
    mu_u = prior_mean + K_uL @ alpha

    # Posterior covariance: Sigma_u = k(X_u, X_u) - k(X_u)^T (sigma^2 I + K_LL)^{-1} k(X_u)
    Sigma_u = K_uu - K_uL @ V
    Sigma_u = 0.5 * (Sigma_u + Sigma_u.T) + jitter * np.eye(n_u)

    return mu_u, Sigma_u


def sample_gp_posterior(
    X_u,
    X_L,
    t_L,
    length_scale=0.15,
    variance=1.5,
    noise_variance=1e-2,
    prior_mean=-3.0,
    jitter=1e-6,
    use_posterior_mean=False,
    rng=None
):
    """
    Draws a spatially correlated latent hazard realization Y_u ~ N(y_bar_u, Sigma_u)
    conditioned on anchor points (X_L, t_L).
    """
    mu_u, Sigma_u = compute_gp_posterior(
        X_u, X_L, t_L,
        length_scale=length_scale,
        variance=variance,
        noise_variance=noise_variance,
        prior_mean=prior_mean,
        jitter=jitter
    )
    if use_posterior_mean:
        return mu_u, mu_u, Sigma_u

    if rng is None:
        Y_u = np.random.multivariate_normal(mu_u, Sigma_u)
    else:
        Y_u = rng.multivariate_normal(mu_u, Sigma_u)

    return Y_u, mu_u, Sigma_u


# ==========================================
# 2. Flattened Invertible Cost Mapping
# ==========================================

def logistic_cost_mapping(y, k_steepness=0.6, y_offset=1.5, max_blockage_prob=None):
    """
    Flattened invertible logistic (sigmoid) mapping C: R -> (0, 1) transforming
    unbounded latent GP dynamics y_e in R into nuanced edge blockage probabilities q_e in [0, 1]:
        q_e = C(y_e) = 1 / (1 + exp(-k_steepness * (y_e - y_offset)))

    With flattened default parameters (k_steepness = 0.6, y_offset = 1.5):
      - At safe anchors (y = -3.0):  q_e = 1 / (1 + exp(-0.6 * (-4.5))) ~ 0.063 (or ~0 near O/D)
      - At neutral baseline (y = 0): q_e = 1 / (1 + exp(-0.6 * (-1.5))) ~ 0.289
      - At flood anchors (y = 2.5):  q_e = 1 / (1 + exp(-0.6 * (1.0)))  ~ 0.646
      - At severe anchors (y = 4.0): q_e = 1 / (1 + exp(-0.6 * (2.5)))  ~ 0.818

    Args:
        y (np.ndarray or float): Unbounded latent GP variables y_e.
        k_steepness (float): Flattened logistic slope parameter (default 0.6).
        y_offset (float): Inflection offset shifting baseline blockage (default 1.5).
        max_blockage_prob (float or None): Optional upper clip bound (e.g., 0.92).

    Returns:
        q (np.ndarray or float): Mapped blockage probabilities in [0, 1].
    """
    y_arr = np.asarray(y, dtype=float)
    q = 1.0 / (1.0 + np.exp(-k_steepness * (y_arr - y_offset)))
    if max_blockage_prob is not None:
        q = np.clip(q, 0.0, max_blockage_prob)
    return float(q) if np.ndim(y) == 0 else q


def inverse_logistic_cost_mapping(q, k_steepness=0.6, y_offset=1.5):
    """
    Exact analytical inverse C^{-1}: (0, 1) -> R (logit function) corresponding to
    the flattened logistic_cost_mapping (Dey et al., 2014):
        y_e = C^{-1}(q_e) = y_offset + (1 / k_steepness) * ln(q_e / (1 - q_e))
    """
    q_clipped = np.clip(np.asarray(q, dtype=float), 1e-9, 1.0 - 1e-9)
    return y_offset + (1.0 / k_steepness) * np.log(q_clipped / (1.0 - q_clipped))


# ==========================================
# 3. Disaster Topology Router & Anchor Builder
# ==========================================

def _get_origin_and_destination_coords(G):
    """
    Extracts the 2D coordinates of the Origin (O) and Destination (D) nodes in G.
    """
    pos = nx.get_node_attributes(G, 'pos')
    nodes = list(G.nodes())
    origins = [n for n, d in G.nodes(data=True) if d.get('is_origin', False)]
    destinations = [n for n, d in G.nodes(data=True) if d.get('is_destination', False)]

    origin_node = origins[0] if origins else min(nodes)
    dest_node = destinations[0] if destinations else max(nodes)

    o_coord = np.asarray(pos[origin_node], dtype=float)
    d_coord = np.asarray(pos[dest_node], dtype=float)
    return origin_node, dest_node, o_coord, d_coord


def _extract_safe_guardian_anchors(G, safe_y=-3.0, ring_radius=0.05):
    """
    Constructs safe-zone anchor coordinates at the Origin and Destination nodes
    (along with a tight local ring of guardian points around O and D) pinned at
    latent intensity y = -3.0 (and below) so the posterior math guarantees ~0%
    blockage probability at the agent's spawn and goal.
    """
    _, _, o_coord, d_coord = _get_origin_and_destination_coords(G)

    # Pin O and D centers and 4 cardinal points immediately around O and D
    angles = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    offsets = np.column_stack([ring_radius * np.cos(angles), ring_radius * np.sin(angles)])

    o_ring = np.clip(o_coord + offsets, 0.0, 1.0)
    d_ring = np.clip(d_coord + offsets, 0.0, 1.0)

    safe_coords = np.vstack([o_coord, d_coord, o_ring, d_ring])
    safe_values = np.full(safe_coords.shape[0], safe_y, dtype=float)
    return safe_coords, safe_values


def _extract_coastal_envelope_anchors(
    G,
    boundary='right',
    num_anchors=8,
    outward_offset=0.025,
    exclude_od_radius=0.22
):
    """
    Places Tsunami coastal anchors strictly on the OUTSIDE perimeter of the graph
    by tracing the graph's outer Convex Hull boundary on the chosen flank and
    pushing the anchor points outward from the hull by `outward_offset`.

    Why Convex Hull:
    - By convexity, 100% of the nodes and edges of G lie strictly inside ConvexHull(V).
    - Therefore, placing anchors along an outward-offset chain of the Convex Hull
      guarantees that NOT A SINGLE ANCHOR can ever lie inside the graph or cut
      through its edges, while keeping the anchors right next to the outermost
      edges and nodes even when the graph is concentrated in the middle of [0, 1]^2.
    """
    pos = nx.get_node_attributes(G, 'pos')
    origin_node, dest_node, o_coord, d_coord = _get_origin_and_destination_coords(G)

    all_pts = np.vstack([np.asarray(pos[n], dtype=float) for n in G.nodes()])
    graph_center = np.mean(all_pts, axis=0)

    hull = ConvexHull(all_pts)
    hull_pts = all_pts[hull.vertices]  # Counterclockwise outer perimeter vertices
    n_h = len(hull_pts)

    # Identify the hull vertices closest to Origin (bottom-left) and Destination (top-right)
    i_o = int(np.argmin(np.linalg.norm(hull_pts - o_coord, axis=1)))
    i_d = int(np.argmin(np.linalg.norm(hull_pts - d_coord, axis=1)))

    # Counterclockwise chain from Origin -> Destination wraps around the Bottom-Right outside of G
    chain_lower_right = []
    curr = i_o
    while True:
        chain_lower_right.append(hull_pts[curr])
        if curr == i_d:
            break
        curr = (curr + 1) % n_h
    chain_lower_right = np.vstack(chain_lower_right)

    # Counterclockwise chain from Destination -> Origin wraps around the Top-Left outside of G
    chain_upper_left = []
    curr = i_d
    while True:
        chain_upper_left.append(hull_pts[curr])
        if curr == i_o:
            break
        curr = (curr + 1) % n_h
    chain_upper_left = np.vstack(chain_upper_left)

    b_side = boundary.strip().lower()
    if b_side in ('right', 'bottom'):
        hull_chain = chain_lower_right
    else:
        hull_chain = chain_upper_left

    # Offset each segment of the hull chain outward away from graph_center
    offset_pts = []
    for k in range(len(hull_chain) - 1):
        p1 = hull_chain[k]
        p2 = hull_chain[k + 1]
        seg = p2 - p1
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6:
            continue
        # Unit normal perpendicular to hull segment
        normal = np.array([seg[1], -seg[0]]) / seg_len
        mid = 0.5 * (p1 + p2)
        if np.dot(normal, mid - graph_center) < 0:
            normal = -normal

        # Sample dense points along this exterior hull segment pushed outward
        t_vals = np.linspace(0.0, 1.0, max(10, int(seg_len * 80)))
        for t in t_vals:
            pt_out = (1.0 - t) * p1 + t * p2 + outward_offset * normal
            pt_out = np.clip(pt_out, 0.01, 0.99)
            offset_pts.append(pt_out)

    offset_pts = np.vstack(offset_pts)

    # Trim points near Origin and Destination so spawn/goal remain completely untouched
    keep_mask = [
        (np.linalg.norm(pt - o_coord) >= exclude_od_radius and np.linalg.norm(pt - d_coord) >= exclude_od_radius)
        for pt in offset_pts
    ]
    valid_pts = offset_pts[keep_mask]
    if len(valid_pts) < num_anchors:
        # Fallback to middle 60% of the exterior hull chain
        n_tot = len(offset_pts)
        valid_pts = offset_pts[int(0.20 * n_tot):max(int(0.80 * n_tot), int(0.20 * n_tot) + 2)]

    # Select `num_anchors` evenly spaced points along the valid exterior hull arc
    idx_sel = np.linspace(0, len(valid_pts) - 1, num_anchors).astype(int)
    hazard_coords = valid_pts[idx_sel]

    # Compute the coastline tangent angle theta so the anisotropic kernel aligns with the coast
    coast_vec = hazard_coords[-1] - hazard_coords[0]
    theta_coast = float(np.arctan2(coast_vec[1], coast_vec[0]))

    # Oriented anisotropic length_scale: (l_parallel_to_coast, l_inland_perp, theta_coast)
    length_scale = (0.22, 0.14, theta_coast)
    return hazard_coords, length_scale


def _extract_cluster_centroids(G, exclude_od_radius=0.22):
    """
    Computes Empirical Cluster Centroids from node 'cluster_id' attributes in G,
    filtering out any clusters too close to the Origin or Destination.
    If G is a single-district graph ('sparse' or 'dense'), partitions intermediate
    nodes into spatial quadrants near the map interior to identify natural hubs.
    """
    pos = nx.get_node_attributes(G, 'pos')
    origin_node, dest_node, o_coord, d_coord = _get_origin_and_destination_coords(G)

    cluster_groups = {}
    for n, attr in G.nodes(data=True):
        if n in (origin_node, dest_node):
            continue
        cid = attr.get('cluster_id', 0)
        cluster_groups.setdefault(cid, []).append(pos[n])

    centroids = []
    if len(cluster_groups) > 1 or (1 in cluster_groups and len(cluster_groups) >= 1):
        for cid, coords_list in sorted(cluster_groups.items()):
            c_mean = np.mean(np.asarray(coords_list, dtype=float), axis=0)
            centroids.append(c_mean)
    else:
        # Fallback for unclustered graphs: select interior candidate hubs
        interior_nodes = [
            np.asarray(pos[n], dtype=float)
            for n in G.nodes()
            if n not in (origin_node, dest_node)
        ]
        if interior_nodes:
            pts = np.vstack(interior_nodes)
            # Sort by distance to center (0.5, 0.5)
            dists_center = np.linalg.norm(pts - np.array([0.5, 0.5]), axis=1)
            top_indices = np.argsort(dists_center)[:min(4, len(pts))]
            centroids = [pts[i] for i in top_indices]
        else:
            centroids = [np.array([0.5, 0.5])]

    centroids = np.vstack(centroids)

    # Filter out centroids that are too close to Origin or Destination
    valid_centroids = []
    for c in centroids:
        if (
            np.linalg.norm(c - o_coord) >= exclude_od_radius
            and np.linalg.norm(c - d_coord) >= exclude_od_radius
        ):
            valid_centroids.append(c)

    if not valid_centroids:
        # Pick the centroid maximizing minimum distance to O and D
        min_od_dists = np.minimum(
            np.linalg.norm(centroids - o_coord, axis=1),
            np.linalg.norm(centroids - d_coord, axis=1)
        )
        best_idx = int(np.argmax(min_od_dists))
        valid_centroids = [centroids[best_idx]]

    return np.vstack(valid_centroids), centroids


def set_disaster_type(
    G,
    disaster_type='earthquake',
    num_epicenters=2,
    boundary='right',
    safe_y=-3.0,
    k_steepness=0.6,
    y_offset=1.5,
    max_blockage_prob=0.92,
    seed=None
):
    """
    Disaster Topology Router that dynamically configures GP posterior anchors (X_L, t_L)
    and kernel hyperparameters based on the requested disaster regime and graph structure:

    1. 'earthquake' (Radial Spike):
       - Extracts 1 or 2 cluster centroids (avoiding Origin and Destination).
       - Sets them as hazard anchors with latent intensity y = 4.0.
       - Uses an isotropic RBF kernel with a short length_scale = 0.10 so damage
         concentrates at the epicenters and dissipates radially.

    2. 'tsunami' (Coastal Sweep):
       - Generates an array of coastal anchor coordinates along one boundary of the
         domain (default x = 1.0, or 'right'/'left'/'top'/'bottom') with y = 4.0,
         filtering out points near O and D.
       - Uses an Anisotropic RBF Kernel (l_x != l_y) with l_y = 0.50 (strong correlation
         parallel to the coast) and l_x = 0.15 (short correlation inland) so the wave
         sweeps inward from the boundary and gradually dissipates.

    3. 'flood' (Meandering River):
       - Generates a smooth meandering spline of anchor coordinates carving through the
         empty valley space between clusters (from top-left/bottom-right orthogonal to O->D)
         with latent intensity y = 2.5.
       - Uses an isotropic RBF kernel with a tight length_scale = 0.05 so the hazard
         strictly follows the riverbanks without flooding the entire domain.

    Crucially, in all regimes, the Origin (O) and Destination (D) nodes are anchored
    with latent intensity y = -3.0 (safe_y) so the GP posterior guarantees a safe
    spawn and goal zone.

    Args:
        G (nx.Graph): Spatial graph generated by SpatialGraphGenerator.
        disaster_type (str): One of {'earthquake', 'tsunami', 'flood'}.
        num_epicenters (int): Number of cluster centroids to target in 'earthquake' (1 or 2).
        boundary (str): Coastal boundary for 'tsunami' ('right' x=1.0, 'left' x=0.0,
                        'top' y=1.0, 'bottom' y=0.0).
        safe_y (float): Latent GP value anchored at Origin and Destination (default -3.0).
        k_steepness (float): Flattened logistic steepness (default 0.6).
        y_offset (float): Flattened logistic offset (default 1.5).
        max_blockage_prob (float): Upper bound on edge blockage probability.
        seed (int or None): Optional random seed.

    Returns:
        G (nx.Graph): Updated graph with 'blockage_prob', 'latent_intensity', and 'gp_state'.
        gp_state (dict): Complete dictionary of GP posterior configuration and sampled state.
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    d_type = disaster_type.strip().lower()

    _, _, o_coord, d_coord = _get_origin_and_destination_coords(G)
    safe_coords, safe_values = _extract_safe_guardian_anchors(G, safe_y=safe_y, ring_radius=0.04)

    if d_type == 'earthquake':
        # Extract 1 or 2 cluster centroids avoiding O and D
        valid_centroids, _ = _extract_cluster_centroids(G, exclude_od_radius=0.22)
        n_select = min(max(1, int(num_epicenters)), len(valid_centroids), 2)
        chosen_idx = rng.choice(len(valid_centroids), size=n_select, replace=False)
        hazard_coords = valid_centroids[chosen_idx]
        hazard_values = np.full(hazard_coords.shape[0], 6.0, dtype=float)

        # Add gentle background far-field anchors at y = -2.0 so ambient map stays clear
        length_scale = 0.10
        variance = 1.5
        noise_variance = 1e-3

    elif d_type == 'tsunami':
        # Place coastal anchors strictly outside the graph's outer Convex Hull
        hazard_coords, length_scale = _extract_coastal_envelope_anchors(
            G,
            boundary=boundary,
            num_anchors=8,
            outward_offset=0.025,
            exclude_od_radius=0.22
        )
        hazard_values = np.full(hazard_coords.shape[0], 4.5, dtype=float)

        variance = 1.5
        noise_variance = 0.25

    elif d_type == 'flood':
        # Generate a meandering river spline carving through the empty space between clusters
        _, all_centroids = _extract_cluster_centroids(G, exclude_od_radius=0.15)

        # Construct a meandering curve across the anti-diagonal [0.1, 0.9] -> [0.9, 0.1]
        # (orthogonal to the O->D diagonal) and deflect waypoints into inter-cluster valleys
        s = np.linspace(0.0, 1.0, 25)
        base_x = 0.10 + 0.80 * s
        base_y = 0.90 - 0.80 * s

        # Sinusoidal meander orthogonal to the river channel
        phase = float(rng.uniform(0.0, np.pi))
        meander_amp = 0.10
        perp_vec = np.array([1.0, 1.0]) / np.sqrt(2.0)

        river_pts = np.column_stack([base_x, base_y]) + (
            meander_amp * np.sin(2.5 * np.pi * s + phase)[:, None] * perp_vec[None, :]
        )

        # Gently push river points away from cluster centroids so the river flows between clusters
        for idx_pt in range(len(river_pts)):
            pt = river_pts[idx_pt]
            for c in all_centroids:
                diff = pt - c
                dist = np.linalg.norm(diff)
                if 1e-4 < dist < 0.14:
                    pt = pt + (diff / dist) * (0.14 - dist) * 0.65
            river_pts[idx_pt] = np.clip(pt, 0.05, 0.95)

        # Smooth the deflected river coordinates with a 3-point moving average spline
        kernel_smooth = np.array([0.25, 0.50, 0.25])
        smooth_x = np.convolve(river_pts[:, 0], kernel_smooth, mode='valid')
        smooth_y = np.convolve(river_pts[:, 1], kernel_smooth, mode='valid')
        river_spline = np.column_stack([smooth_x, smooth_y])

        # Filter out any river points too close to Origin or Destination
        keep_mask = [
            (np.linalg.norm(pt - o_coord) >= 0.16 and np.linalg.norm(pt - d_coord) >= 0.16)
            for pt in river_spline
        ]
        hazard_coords = river_spline[keep_mask]
        if len(hazard_coords) == 0:
            hazard_coords = river_spline
        hazard_values = np.full(hazard_coords.shape[0], 2.5, dtype=float)

        length_scale = 0.05
        variance = 1.5
        noise_variance = 1e-3

    else:
        raise ValueError(
            f"Unsupported disaster_type '{disaster_type}'. Choose from 'earthquake', 'tsunami', or 'flood'."
        )

    # Combine hazard anchors and Origin/Destination safe-zone anchors (y = -3.0)
    anchor_coords = np.vstack([hazard_coords, safe_coords])
    anchor_values = np.concatenate([hazard_values, safe_values])

    gp_params = {
        'disaster_type': d_type,
        'length_scale': length_scale,
        'variance': variance,
        'noise_variance': noise_variance,
        'k_steepness': k_steepness,
        'y_offset': y_offset,
        'max_blockage_prob': max_blockage_prob,
        'anchor_coords': anchor_coords,
        'anchor_values': anchor_values,
        'hazard_anchor_coords': hazard_coords,
        'safe_anchor_coords': safe_coords[:2],  # [Origin, Destination]
        'safe_y': safe_y,
        'seed': seed,
    }

    return apply_gp_disaster_to_graph(G, gp_params)


# ==========================================
# 4. Graph Edge Integration & Realization
# ==========================================

def extract_edge_midpoints(G):
    """
    Extracts the ordered list of edges and their 2D spatial midpoints
    m_e = ((x_u + x_v) / 2, (y_u + y_v) / 2) for all e = (u, v) in G.

    Returns:
        edge_list (list of tuple): Ordered list of edges (u, v).
        midpoints (np.ndarray): Array of shape (|E|, 2) with midpoint coordinates.
    """
    pos = nx.get_node_attributes(G, 'pos')
    edge_list = list(G.edges())
    midpoints = np.zeros((len(edge_list), 2), dtype=float)

    for idx, (u, v) in enumerate(edge_list):
        x_u, y_u = pos[u]
        x_v, y_v = pos[v]
        midpoints[idx, 0] = 0.5 * (x_u + x_v)
        midpoints[idx, 1] = 0.5 * (y_u + y_v)

    return edge_list, midpoints


def apply_gp_disaster_to_graph(G, gp_params=None):
    """
    Evaluates the Gaussian Process over the spatial midpoints of all edges in G.
    Always anchors the Origin and Destination nodes at y = -3.0 (unless custom
    anchors are already supplied in gp_params) so the posterior distribution
        y_bar_u = k(x_u)^T (sigma^2 I + K_LL)^{-1} t_L
        Sigma_u = k(x_u, x_u) - k(x_u)^T (sigma^2 I + K_LL)^{-1} k(x_u)
    guarantees safe spawn and goal zones. Maps latent intensities y_e to blockage
    probabilities q_e via the flattened logistic function.

    Args:
        G (nx.Graph): Spatial graph generated by SpatialGraphGenerator.
        gp_params (dict or None): GP kernel, anchor, and logistic mapping parameters.

    Returns:
        G (nx.Graph): Graph updated with 'blockage_prob', 'latent_intensity', and 'posterior_mean'.
        gp_state (dict): Complete dictionary of GP state for downstream visualization.
    """
    if gp_params is None:
        gp_params = {}

    disaster_type = gp_params.get('disaster_type', 'custom')
    length_scale = gp_params.get('length_scale', 0.15)
    variance = gp_params.get('variance', 1.5)
    noise_variance = gp_params.get('noise_variance', 1e-3)
    k_steepness = gp_params.get('k_steepness', 0.6)
    y_offset = gp_params.get('y_offset', 1.5)
    max_blockage_prob = gp_params.get('max_blockage_prob', 0.92)
    safe_y = gp_params.get('safe_y', -3.0)
    use_posterior_mean = gp_params.get('use_posterior_mean', True)
    jitter = gp_params.get('jitter', 1e-6)
    seed = gp_params.get('seed', None)

    rng = np.random.default_rng(seed) if seed is not None else None

    # 1. Extract edge midpoints X_u
    edge_list, midpoints = extract_edge_midpoints(G)
    if len(edge_list) == 0:
        raise ValueError("Graph G contains no edges.")

    # 2. Ensure Origin and Destination safe anchors (y = -3.0) are always present
    origin_node, dest_node, o_coord, d_coord = _get_origin_and_destination_coords(G)
    anchor_coords = gp_params.get('anchor_coords', None)
    anchor_values = gp_params.get('anchor_values', None)

    if anchor_coords is None or anchor_values is None:
        safe_coords, safe_values = _extract_safe_guardian_anchors(G, safe_y=safe_y, ring_radius=0.04)
        anchor_coords = safe_coords
        anchor_values = safe_values
    else:
        anchor_coords = np.atleast_2d(np.asarray(anchor_coords, dtype=float))
        anchor_values = np.asarray(anchor_values, dtype=float).ravel()
        # Verify Origin and Destination are anchored at safe_y
        dist_o = np.min(np.linalg.norm(anchor_coords - o_coord, axis=1))
        dist_d = np.min(np.linalg.norm(anchor_coords - d_coord, axis=1))
        if dist_o > 1e-3 or dist_d > 1e-3:
            safe_coords, safe_values = _extract_safe_guardian_anchors(G, safe_y=safe_y, ring_radius=0.04)
            anchor_coords = np.vstack([anchor_coords, safe_coords])
            anchor_values = np.concatenate([anchor_values, safe_values])

    # 3. Compute GP Posterior mean y_bar_u and covariance Sigma_u at edge midpoints
    y_edges, mu_edges, Sigma_edges = sample_gp_posterior(
        X_u=midpoints,
        X_L=anchor_coords,
        t_L=anchor_values,
        length_scale=length_scale,
        variance=variance,
        noise_variance=noise_variance,
        prior_mean=safe_y,
        jitter=jitter,
        use_posterior_mean=use_posterior_mean,
        rng=rng
    )

    # 4. Map latent intensities y_e -> blockage probabilities q_e via flattened logistic C(y_e)
    q_edges = logistic_cost_mapping(
        y_edges,
        k_steepness=k_steepness,
        y_offset=y_offset,
        max_blockage_prob=max_blockage_prob
    )

    # Guarantee edges directly incident to Origin or Destination have near-zero blockage
    # consistent with the y = -3.0 safe anchor condition
    safe_prob_floor = logistic_cost_mapping(safe_y, k_steepness=k_steepness, y_offset=y_offset)

    for idx, (u, v) in enumerate(edge_list):
        q_val = float(q_edges[idx])
        if u in (origin_node, dest_node) or v in (origin_node, dest_node):
            # Smoothly cap incident spawn/goal edges so the agent is never trapped at step 0
            q_val = min(q_val, float(safe_prob_floor))

        G.edges[u, v]['midpoint'] = (float(midpoints[idx, 0]), float(midpoints[idx, 1]))
        G.edges[u, v]['latent_intensity'] = float(y_edges[idx])
        G.edges[u, v]['posterior_mean'] = float(mu_edges[idx])
        G.edges[u, v]['posterior_std'] = float(np.sqrt(max(0.0, Sigma_edges[idx, idx])))
        G.edges[u, v]['blockage_prob'] = q_val

    gp_state = dict(gp_params)
    gp_state.update({
        'disaster_type': disaster_type,
        'length_scale': length_scale,
        'variance': variance,
        'noise_variance': noise_variance,
        'k_steepness': k_steepness,
        'y_offset': y_offset,
        'max_blockage_prob': max_blockage_prob,
        'safe_y': safe_y,
        'jitter': jitter,
        'anchor_coords': anchor_coords,
        'anchor_values': anchor_values,
        'edge_midpoints': midpoints,
        'edge_latent_y': y_edges,
        'edge_posterior_mu': mu_edges,
        'edge_posterior_cov': Sigma_edges,
        'edge_blockage_q': np.array([G.edges[u, v]['blockage_prob'] for u, v in edge_list]),
    })

    G.graph['gp_state'] = gp_state
    return G, gp_state


def sample_gp_disaster_realization(G, max_resamples=1000, allow_relaxation=True):
    """
    Draws Bernoulli trials B_e ~ Bern(q_e) for every edge in G based on the GP-derived
    blockage probabilities q_e. Verifies that an Origin -> Destination path exists
    in G_offline, using multi-stage relaxation and a safest-path fallback if needed.

    Returns:
        G_offline (nx.Graph): Graph with blocked edges removed.
        blocked_edges (list): List of edges (u, v) that failed in this realization.
    """
    nodes = list(G.nodes())
    origins = [n for n, d in G.nodes(data=True) if d.get('is_origin', False)]
    destinations = [n for n, d in G.nodes(data=True) if d.get('is_destination', False)]

    origin = origins[0] if origins else min(nodes)
    destination = destinations[0] if destinations else max(nodes)

    scale = 1.0
    max_relax_stages = 6 if allow_relaxation else 1
    attempts_per_stage = max(1, max_resamples // max_relax_stages) if allow_relaxation else max_resamples

    for _ in range(max_relax_stages):
        for _ in range(attempts_per_stage):
            G_offline = G.copy()
            blocked_edges = []

            for u, v, data in G.edges(data=True):
                q_e = data.get('blockage_prob', 0.0) * scale
                is_blocked = (np.random.rand() < q_e)
                G.edges[u, v]['is_blocked'] = int(is_blocked)

                if is_blocked:
                    blocked_edges.append((u, v))
                    G_offline.remove_edge(u, v)

            if nx.has_path(G_offline, origin, destination):
                return G_offline, blocked_edges

        scale *= 0.80

    if allow_relaxation and nx.has_path(G, origin, destination):
        def edge_risk_weight(u, v, d):
            q = min(0.999, d.get('blockage_prob', 0.0))
            return -np.log(1.0 - q)

        safest_path = nx.shortest_path(G, origin, destination, weight=edge_risk_weight)
        protected_edges = {
            (min(safest_path[k], safest_path[k + 1]), max(safest_path[k], safest_path[k + 1]))
            for k in range(len(safest_path) - 1)
        }

        G_offline = G.copy()
        blocked_edges = []
        for u, v, data in G.edges(data=True):
            edge_key = (min(u, v), max(u, v))
            if edge_key in protected_edges:
                is_blocked = False
            else:
                q_e = data.get('blockage_prob', 0.0)
                is_blocked = (np.random.rand() < q_e)

            G.edges[u, v]['is_blocked'] = int(is_blocked)
            if is_blocked:
                blocked_edges.append((u, v))
                G_offline.remove_edge(u, v)

        return G_offline, blocked_edges

    raise RuntimeError(
        f"Failed to sample an O->D connected G_offline after {max_resamples} attempts."
    )


# ==========================================
# 5. Continuous Meshgrid GP Evaluation & Visualization
# ==========================================

def evaluate_gp_on_meshgrid(grid_coords, gp_state):
    """
    Evaluates the continuous 2D hazard field over meshgrid coordinates X_grid
    using exact GP Posterior Conditioning (kriging) from the anchor observations
    (X_L, t_L):
        y_bar_grid = m_0 + k(X_grid)^T (sigma^2 I + K_LL)^{-1} (t_L - m_0)
    """
    length_scale = gp_state.get('length_scale', 0.15)
    variance = gp_state.get('variance', 1.5)
    noise_variance = gp_state.get('noise_variance', 1e-3)
    prior_mean = gp_state.get('safe_y', -3.0)
    jitter = gp_state.get('jitter', 1e-6)

    X_L = gp_state.get('anchor_coords')
    t_L = gp_state.get('anchor_values')

    if X_L is not None and t_L is not None:
        X_L = np.atleast_2d(np.asarray(X_L, dtype=float))
        t_L = np.asarray(t_L, dtype=float).ravel()
        K_LL = rbf_kernel(X_L, X_L, length_scale=length_scale, variance=variance)
        K_LL_reg = K_LL + (noise_variance + jitter) * np.eye(len(X_L))
        K_gL = rbf_kernel(grid_coords, X_L, length_scale=length_scale, variance=variance)
        alpha = np.linalg.solve(K_LL_reg, t_L - prior_mean)
        return prior_mean + K_gL @ alpha

    # Fallback if only edge midpoints are present
    X_edges = gp_state.get('edge_midpoints')
    y_edges = gp_state.get('edge_latent_y')
    if X_edges is not None and y_edges is not None:
        K_ee = rbf_kernel(X_edges, X_edges, length_scale=length_scale, variance=variance)
        K_ee_reg = K_ee + jitter * np.eye(len(X_edges))
        K_ge = rbf_kernel(grid_coords, X_edges, length_scale=length_scale, variance=variance)
        alpha = np.linalg.solve(K_ee_reg, y_edges - prior_mean)
        return prior_mean + K_ge @ alpha

    y_grid, _ = sample_gp_prior(grid_coords, length_scale=length_scale, variance=variance, jitter=jitter)
    return y_grid


def visualize_gaussian_disaster(G, gp_params=None, show_labels=True, grid_res=100):
    """
    Renders the spatial network overlaid on the continuous 2D Gaussian Process
    posterior hazard field heatmap:
    - Background: Continuous GP-conditioned blockage probability field C(y_bar(x))
      over [0, 1]^2 rendered via matplotlib.pyplot.contourf with 'RdYlGn_r'.
    - Anchors: Plots disaster anchor locations (epicenters, coastal wave line, or river spline).
    - Nodes: Origin (O) as a Star ('*'), Destination (D) as an 'X', and intermediate nodes.
    - Edges: Color-coded using a 'YlOrRd' gradient based on their blockage probability q_e,
      with numerical q_e text labels annotated alongside each edge.
    """
    gp_state = G.graph.get('gp_state', {})
    if gp_params is not None:
        merged_params = dict(gp_state)
        merged_params.update(gp_params)
        gp_state = merged_params

    if len(G.edges()) > 0 and 'blockage_prob' not in next(iter(G.edges(data=True)))[2]:
        G, gp_state = apply_gp_disaster_to_graph(G, gp_state)

    disaster_type = gp_state.get('disaster_type', 'custom').capitalize()
    k_steepness = gp_state.get('k_steepness', 0.6)
    y_offset = gp_state.get('y_offset', 1.5)
    max_blockage_prob = gp_state.get('max_blockage_prob', 0.92)
    length_scale = gp_state.get('length_scale', 0.15)
    variance = gp_state.get('variance', 1.5)

    ls_arr = np.asarray(length_scale, dtype=float).ravel()
    if ls_arr.size == 1:
        ls_str = f"l={float(ls_arr[0]):.2f}"
    elif ls_arr.size == 2:
        ls_str = f"l_x={ls_arr[0]:.2f}, l_y={ls_arr[1]:.2f}"
    else:
        ls_str = f"l_\\parallel={ls_arr[0]:.2f}, l_\\perp={ls_arr[1]:.2f}"

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_title(
        f"GP Posterior Disaster Field [{disaster_type}] "
        f"(${ls_str}, \\sigma_f^2={variance:.2f}, k={k_steepness:.2f}, y_0={y_offset:.2f}$)",
        fontsize=14,
        fontweight='bold'
    )

    # 1. Create 2D Meshgrid over [0, 1]^2 and evaluate GP posterior projection
    gx = np.linspace(0.0, 1.0, grid_res)
    gy = np.linspace(0.0, 1.0, grid_res)
    X, Y = np.meshgrid(gx, gy)
    grid_coords = np.column_stack([X.ravel(), Y.ravel()])

    y_grid = evaluate_gp_on_meshgrid(grid_coords, gp_state)
    q_grid = logistic_cost_mapping(
        y_grid,
        k_steepness=k_steepness,
        y_offset=y_offset,
        max_blockage_prob=max_blockage_prob
    ).reshape(grid_res, grid_res)

    # Render continuous background contour plot using 'RdYlGn_r'
    bg_cmap = mpl.colormaps['RdYlGn_r']
    contour = ax.contourf(
        X, Y, q_grid,
        levels=50,
        cmap=bg_cmap,
        alpha=0.58,
        vmin=0.0,
        vmax=1.0
    )
    cbar_bg = fig.colorbar(contour, ax=ax, fraction=0.035, pad=0.02)
    cbar_bg.set_label("Posterior Hazard Probability $C(\\bar{y}(\\mathbf{x}))$", fontsize=10)

    # Plot Hazard Anchor Markers if available
    hazard_anchors = gp_state.get('hazard_anchor_coords')
    if hazard_anchors is not None and len(hazard_anchors) > 0:
        hazard_anchors = np.atleast_2d(hazard_anchors)
        ax.scatter(
            hazard_anchors[:, 0], hazard_anchors[:, 1],
            s=90, facecolors='none', edgecolors='black',
            linewidths=1.8, linestyle='--', zorder=6, label='Hazard Anchors ($t_L > 0$)'
        )

    # 2. Extract Node Positions & Categories
    pos = nx.get_node_attributes(G, 'pos')
    origins = [n for n, attr in G.nodes(data=True) if attr.get('is_origin')]
    destinations = [n for n, attr in G.nodes(data=True) if attr.get('is_destination')]
    standard_nodes = [n for n in G.nodes() if n not in origins and n not in destinations]

    # 3. Draw Edges Color-Coded by Blockage Probability q_e ('YlOrRd')
    edge_cmap = mpl.colormaps['YlOrRd']
    edge_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    segments = []
    q_values = []
    linewidths = []

    for u, v, attr in G.edges(data=True):
        segments.append([pos[u], pos[v]])
        q_e = attr.get('blockage_prob', 0.0)
        q_values.append(q_e)
        linewidths.append(2.8 if attr.get('is_highway', 0) == 1 else 1.8)

    lc = LineCollection(
        segments,
        cmap=edge_cmap,
        norm=edge_norm,
        linewidths=linewidths,
        alpha=0.95,
        zorder=3
    )
    lc.set_array(np.array(q_values))
    ax.add_collection(lc)

    cbar_edge = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.08)
    cbar_edge.set_label("Edge Blockage Probability $q_e$", fontsize=10)

    # 4. Annotate Edge Blockage Probabilities (q_e)
    if show_labels:
        edge_labels = {
            (u, v): f"{attr.get('blockage_prob', 0.0):.2f}"
            for u, v, attr in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=7,
            font_color='black',
            font_weight='bold',
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.0),
            ax=ax
        )

    # 5. Overlay Nodes (Origin as Star, Destination as X)
    nx.draw_networkx_nodes(
        G, pos, nodelist=standard_nodes, node_size=140,
        node_color='white', edgecolors='black', linewidths=1.2, ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=origins, node_size=380,
        node_color='dodgerblue', node_shape='*', edgecolors='black', linewidths=1.5, ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=destinations, node_size=320,
        node_color='crimson', node_shape='X', edgecolors='black', linewidths=1.5, ax=ax
    )
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', ax=ax)

    # 6. Plot Formatting & Legend
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.35)

    legend_handles = [
        Line2D([0], [0], marker='*', color='w', label='Origin (O) [Safe Anchor $y=-3.0$]',
               markerfacecolor='dodgerblue', markersize=14, markeredgecolor='k'),
        Line2D([0], [0], marker='X', color='w', label='Destination (D) [Safe Anchor $y=-3.0$]',
               markerfacecolor='crimson', markersize=12, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Intermediate Node',
               markerfacecolor='white', markersize=8, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Disaster Anchor ($t_L$)',
               markerfacecolor='none', markersize=9, markeredgecolor='k', markeredgewidth=1.8),
    ]
    ax.legend(handles=legend_handles, loc='upper left', framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.show()


# ==========================================
# 6. Example Execution Script
# ==========================================
if __name__ == "__main__":
    from graph_builder import SpatialGraphGenerator

    # 1. Build a synthetic multi-district spatial network
    builder = SpatialGraphGenerator(n_nodes=40, d_min=0.04)
    G_base, _ = builder.build_graph(regime='multidistrict', C=6)

    # 2. Demonstrate all three structured GP Posterior Disaster Topologies
    for mode in ['earthquake', 'tsunami', 'flood']:
        G_sim = G_base.copy()
        G_sim, gp_state = set_disaster_type(
            G_sim,
            disaster_type=mode,
            num_epicenters=1,
            boundary='right',
            safe_y=-3.0,
            k_steepness=0.6,
            y_offset=1.5,
        )
        G_offline, blocked_edges = sample_gp_disaster_realization(G_sim)
        print(
            f"[{mode.upper()}] Total Edges: {G_sim.number_of_edges()} | "
            f"Blocked in Realization: {len(blocked_edges)} | "
            f"Remaining in G_offline: {G_offline.number_of_edges()}"
        )
        visualize_gaussian_disaster(G_sim, gp_state, show_labels=True)
