#!/usr/bin/env python3
"""
convert_all.py — Stage 1 Pre-processing Pipeline
=================================================
Converts a SimVascular project (0074_H_CORO_KD) into PyG Data objects.

Data sources:
  - Paths/*.pth          → node coordinates (x, y, z)
  - Segmentations/*.ctgr → vessel radii per cross-section
  - COR_Pim_*            → outlet pressure waveforms (labels)
  - inflow_1d.flow       → inlet flow rate boundary condition
  - bct.vtp              → inlet velocity boundary condition

Graph construction:
  - Each path control point → graph node
  - Consecutive points on same path → edge (bidirectional)
  - Junction nodes (closest points between paths) → cross-path edges
  - Node features x:        [N, 5]  (x, y, z, radius, is_junction)
  - Edge features edge_attr: [E, 2]  (avg_radius, segment_length)
  - Labels y:               [N, 2]  (pressure, velocity_magnitude)

MeshGraphNet interface:
  model(node_features, edge_features, edge_index)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    from torch_geometric.data import Data
except ImportError:
    raise ImportError("pip install torch_geometric")

try:
    import pyvista as pv
except ImportError:
    pv = None  # optional, only needed for bct.vtp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================
# Data classes
# ======================================================================
@dataclass
class VesselPath:
    """A single vessel centerline path."""
    name: str
    points: np.ndarray          # (M, 3) control points
    radii: np.ndarray           # (M,)   interpolated radius per point
    path_id: Optional[int] = None


@dataclass
class ContourGroup:
    """Cross-section contours for one vessel segment."""
    path_name: str
    centers: np.ndarray         # (K, 3)
    radii: np.ndarray           # (K,)


@dataclass
class PressureWaveform:
    """Time-pressure data from a COR_Pim file."""
    name: str
    time: np.ndarray            # (T,)
    pressure: np.ndarray        # (T,)


# ======================================================================
# Parsers
# ======================================================================
def parse_sv_xml(filepath: Path) -> ET.Element:
    """Parse SimVascular XML files that have multiple root elements."""
    content = filepath.read_text(encoding="utf-8")
    # Strip XML declaration and standalone tags
    content = content.replace('<?xml version="1.0" encoding="UTF-8" ?>', "")
    content = content.replace('<?xml version="1.0" encoding="utf-8"?>', "")
    content = content.replace('<format version="1.0" />', "")
    return ET.fromstring("<root>" + content.strip() + "</root>")


def parse_path(pth_file: Path) -> VesselPath:
    """Parse a SimVascular .pth file → VesselPath."""
    tree = parse_sv_xml(pth_file)
    pts_elems = tree.findall(".//control_points/point")
    coords = np.array(
        [[float(p.get("x")), float(p.get("y")), float(p.get("z"))]
         for p in pts_elems],
        dtype=np.float32,
    )
    name = pth_file.stem
    path_id_elem = tree.find(".//path")
    path_id = int(path_id_elem.get("id")) if path_id_elem is not None else None

    return VesselPath(
        name=name,
        points=coords,
        radii=np.zeros(len(coords), dtype=np.float32),  # filled later
        path_id=path_id,
    )


def parse_contour_group(ctgr_file: Path) -> ContourGroup:
    """Parse a SimVascular .ctgr file → ContourGroup with centers and radii."""
    tree = parse_sv_xml(ctgr_file)
    contours = tree.findall(".//contour")

    centers = []
    radii = []

    for c in contours:
        pos = c.find("path_point/pos")
        cpts = c.findall("contour_points/point")
        if pos is None or not cpts:
            continue

        cx = float(pos.get("x"))
        cy = float(pos.get("y"))
        cz = float(pos.get("z"))
        centers.append([cx, cy, cz])

        # Radius = average distance from center to contour points
        dists = []
        for pt in cpts:
            dx = float(pt.get("x")) - cx
            dy = float(pt.get("y")) - cy
            dz = float(pt.get("z")) - cz
            dists.append(math.sqrt(dx * dx + dy * dy + dz * dz))
        radii.append(sum(dists) / len(dists))

    return ContourGroup(
        path_name=ctgr_file.stem,
        centers=np.array(centers, dtype=np.float32) if centers else np.zeros((0, 3), dtype=np.float32),
        radii=np.array(radii, dtype=np.float32),
    )


def parse_pressure_waveform(filepath: Path) -> PressureWaveform:
    """Parse a COR_Pim_* file → PressureWaveform."""
    data = np.loadtxt(str(filepath))
    return PressureWaveform(
        name=filepath.name,
        time=data[:, 0].astype(np.float32),
        pressure=data[:, 1].astype(np.float32),
    )


def parse_inflow(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse inflow_1d.flow → (time, flow_rate)."""
    data = np.loadtxt(str(filepath))
    return data[:, 0].astype(np.float32), data[:, 1].astype(np.float32)


# ======================================================================
# Radius interpolation
# ======================================================================
def assign_radii_to_path(path: VesselPath, contour_groups: list[ContourGroup]) -> None:
    """
    Match contour groups to a path by name prefix, then interpolate
    radii onto path control points via nearest-neighbor mapping.
    """
    # Collect all contour centers+radii that belong to this path
    # Matching logic: "aorta.ctgr" matches path "aorta"
    #                 "left(2).ctgr", "left(3).ctgr" match path "left"
    #                 "right(16).ctgr" matches path "right"
    #                 "L(51).ctgr" matches path "L"
    #                 "top(2).ctgr" matches path "top"
    all_centers = []
    all_radii = []

    for cg in contour_groups:
        # Extract base name: "left(2)" → "left", "aorta" → "aorta"
        base = cg.path_name.split("(")[0]
        if base.lower() == path.name.lower():
            if len(cg.centers) > 0:
                all_centers.append(cg.centers)
                all_radii.append(cg.radii)

    if not all_centers:
        logger.warning("  ⚠ No contour match for path '%s', using default radius 0.5", path.name)
        path.radii = np.full(len(path.points), 0.5, dtype=np.float32)
        return

    centers = np.concatenate(all_centers, axis=0)  # (K, 3)
    radii = np.concatenate(all_radii, axis=0)      # (K,)

    # For each path point, find nearest contour center and assign its radius
    for i, pt in enumerate(path.points):
        dists = np.linalg.norm(centers - pt, axis=1)
        nearest_idx = np.argmin(dists)
        path.radii[i] = radii[nearest_idx]

    logger.info(
        "  → Path '%s': radius range [%.3f, %.3f], mean=%.3f",
        path.name, path.radii.min(), path.radii.max(), path.radii.mean(),
    )


# ======================================================================
# Graph construction
# ======================================================================
def find_junction_pairs(
    paths: list[VesselPath], threshold: float = 2.0
) -> list[tuple[int, int, int, int]]:
    """
    Find junction points between different paths.
    Returns list of (path_i, point_i, path_j, point_j) where endpoints
    of one path are close to points on another path.

    For coronary anatomy, we expect:
      - top connects to aorta (ascending aorta)
      - L (LCA main) connects to aorta
      - left branches connect to L
      - right (RCA) connects to aorta
    """
    junctions = []

    for i, pi in enumerate(paths):
        for j, pj in enumerate(paths):
            if i >= j:
                continue
            # Check all endpoints of pi against all points of pj
            for pi_idx in [0, len(pi.points) - 1]:  # endpoints only
                pt = pi.points[pi_idx]
                dists = np.linalg.norm(pj.points - pt, axis=1)
                min_idx = np.argmin(dists)
                min_dist = dists[min_idx]
                if min_dist < threshold:
                    junctions.append((i, pi_idx, j, int(min_idx)))
                    logger.info(
                        "  → Junction: %s[%d] ↔ %s[%d] (dist=%.3f)",
                        pi.name, pi_idx, pj.name, min_idx, min_dist,
                    )

    return junctions


def build_graph(
    paths: list[VesselPath],
    contour_groups: list[ContourGroup],
    junction_threshold: float = 2.0,
) -> Data:
    """
    Build a PyG graph from vessel paths.

    Node features:  [N, 5] = (x, y, z, radius, is_junction)
    Edge features:  [E, 2] = (avg_radius, segment_length)
    Edge index:     [2, E] = bidirectional connectivity
    """
    # --- Assign radii ---
    for path in paths:
        assign_radii_to_path(path, contour_groups)

    # --- Build global node list ---
    # Each path contributes its control points as nodes
    # Track: global_offset[i] = starting global index for path i
    global_offset = []
    all_coords = []
    all_radii = []
    offset = 0

    for path in paths:
        global_offset.append(offset)
        all_coords.append(path.points)
        all_radii.append(path.radii)
        offset += len(path.points)

    coords = np.concatenate(all_coords, axis=0)   # (N, 3)
    radii = np.concatenate(all_radii, axis=0)      # (N,)
    N = len(coords)
    is_junction = np.zeros(N, dtype=np.float32)

    # --- Build intra-path edges (consecutive points) ---
    src_list, dst_list = [], []

    for i, path in enumerate(paths):
        off = global_offset[i]
        for k in range(len(path.points) - 1):
            a = off + k
            b = off + k + 1
            # Bidirectional
            src_list.extend([a, b])
            dst_list.extend([b, a])

    # --- Build inter-path edges (junctions) ---
    junctions = find_junction_pairs(paths, threshold=junction_threshold)
    for pi_idx, pi_pt, pj_idx, pj_pt in junctions:
        a = global_offset[pi_idx] + pi_pt
        b = global_offset[pj_idx] + pj_pt
        src_list.extend([a, b])
        dst_list.extend([b, a])
        is_junction[a] = 1.0
        is_junction[b] = 1.0

    src = np.array(src_list, dtype=np.int64)
    dst = np.array(dst_list, dtype=np.int64)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    num_edges = edge_index.shape[1]

    # --- Edge features ---
    edge_radii = np.zeros(num_edges, dtype=np.float32)
    edge_lengths = np.zeros(num_edges, dtype=np.float32)
    for e in range(num_edges):
        s, d = int(src[e]), int(dst[e])
        edge_radii[e] = 0.5 * (radii[s] + radii[d])
        edge_lengths[e] = float(np.linalg.norm(coords[s] - coords[d]))

    edge_attr = torch.tensor(
        np.stack([edge_radii, edge_lengths], axis=-1), dtype=torch.float32
    )  # (E, 2)

    # --- Node features ---
    x = torch.tensor(
        np.column_stack([coords, radii, is_junction]), dtype=torch.float32
    )  # (N, 5)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = N

    return data, global_offset


# ======================================================================
# Label assignment
# ======================================================================
def assign_labels(
    data: Data,
    paths: list[VesselPath],
    global_offset: list[int],
    pressure_waveforms: list[PressureWaveform],
    inflow_time: np.ndarray,
    inflow_rate: np.ndarray,
    time_index: int = 500,  # peak systole (~t=0.5)
) -> None:
    """
    Assign pressure and velocity labels to each node.

    Strategy:
    - Pressure: Map COR_Pim waveforms to outlet nodes.
      COR_Pim_6~9 → aorta outlets (higher P range)
      COR_Pim_10~14 → coronary outlets (lower P range)
      Interior nodes: linear interpolation between inlet and outlet.
    - Velocity: Derive from flow rate Q and cross-section area A=πr².
      v = Q / A (1D assumption)

    Labels y: [N, 2] = (pressure_at_time_t, velocity_magnitude)
    """
    N = data.num_nodes
    pressure = np.zeros(N, dtype=np.float32)
    velocity = np.zeros(N, dtype=np.float32)

    # --- Pressure assignment ---
    # Get representative pressures at chosen time index
    if pressure_waveforms:
        # Sort waveforms: COR_Pim_6~9 (aorta), COR_Pim_10~14 (coronary)
        aorta_pws = [pw for pw in pressure_waveforms
                     if int(pw.name.split("_")[-1]) <= 9]
        coro_pws = [pw for pw in pressure_waveforms
                    if int(pw.name.split("_")[-1]) >= 10]

        # Representative pressure at chosen time
        t_idx = min(time_index, len(pressure_waveforms[0].pressure) - 1)

        p_aorta_outlet = (
            np.mean([pw.pressure[t_idx] for pw in aorta_pws])
            if aorta_pws else 0.0
        )
        p_coro_outlet = (
            np.mean([pw.pressure[t_idx] for pw in coro_pws])
            if coro_pws else 0.0
        )

        logger.info(
            "  Pressure at t_idx=%d: aorta_outlet=%.0f, coro_outlet=%.0f dyne/cm²",
            t_idx, p_aorta_outlet, p_coro_outlet,
        )

        # Assign to paths based on anatomy
        # Mapping: path name → pressure regime
        path_pressure_map = {
            "top": ("inlet", None),         # ascending aorta (inlet side)
            "aorta": ("aorta", p_aorta_outlet),
            "L": ("coro", p_coro_outlet),   # LCA main stem
            "left": ("coro", p_coro_outlet),
            "right": ("coro", p_coro_outlet),
        }

        # Inlet pressure estimate (higher than outlets due to pressure drop)
        p_inlet = max(p_aorta_outlet, p_coro_outlet) * 1.05

        for i, path in enumerate(paths):
            off = global_offset[i]
            n_pts = len(path.points)
            regime, p_outlet = path_pressure_map.get(
                path.name, ("coro", p_coro_outlet)
            )

            if regime == "inlet":
                # Ascending aorta: near inlet pressure
                pressure[off : off + n_pts] = p_inlet
            else:
                # Linear gradient from inlet-side to outlet-side
                for k in range(n_pts):
                    frac = k / max(n_pts - 1, 1)  # 0 at start, 1 at end
                    pressure[off + k] = p_inlet * (1 - frac) + p_outlet * frac

    # --- Velocity assignment ---
    # v = Q / A = Q / (π r²)
    if len(inflow_rate) > 0:
        q_idx = min(time_index, len(inflow_rate) - 1)
        Q = inflow_rate[q_idx]  # total flow rate at this time

        radii_np = data.x[:, 3].numpy()  # radius per node
        areas = np.pi * radii_np ** 2
        # Avoid division by zero
        areas = np.clip(areas, 1e-6, None)
        velocity = Q / areas
        # Cap extreme values (very small vessels)
        velocity = np.clip(velocity, 0, np.percentile(velocity, 99))

    data.y = torch.tensor(
        np.column_stack([pressure, velocity]), dtype=torch.float32
    )  # (N, 2)


# ======================================================================
# Time-series dataset (multiple snapshots)
# ======================================================================
def build_temporal_dataset(
    paths: list[VesselPath],
    contour_groups: list[ContourGroup],
    pressure_waveforms: list[PressureWaveform],
    inflow_time: np.ndarray,
    inflow_rate: np.ndarray,
    n_snapshots: int = 20,
    junction_threshold: float = 2.0,
) -> list[Data]:
    """
    Build multiple graph snapshots at different time points.
    Each snapshot shares the same topology but different P/v labels.
    """
    dataset = []

    # Build graph structure once
    base_data, global_offset = build_graph(
        paths, contour_groups, junction_threshold
    )

    # Determine time indices to sample
    if pressure_waveforms:
        n_timesteps = len(pressure_waveforms[0].pressure)
    else:
        n_timesteps = len(inflow_rate) if len(inflow_rate) > 0 else 100

    step = max(1, n_timesteps // n_snapshots)
    time_indices = list(range(0, n_timesteps, step))[:n_snapshots]

    logger.info("Building %d temporal snapshots from %d total timesteps",
                len(time_indices), n_timesteps)

    for t_idx in time_indices:
        # Normalized time feature: t_norm ∈ [0, 1] across the full waveform
        t_norm = t_idx / max(n_timesteps - 1, 1)
        t_col = torch.full(
            (base_data.num_nodes, 1), t_norm, dtype=torch.float32
        )  # (N, 1)

        # Clone the base graph and append time as 6th node feature
        snapshot = Data(
            x=torch.cat([base_data.x.clone(), t_col], dim=1),  # (N, 6)
            edge_index=base_data.edge_index.clone(),
            edge_attr=base_data.edge_attr.clone(),
        )
        snapshot.num_nodes = base_data.num_nodes

        # Assign labels at this time index
        assign_labels(
            snapshot, paths, global_offset,
            pressure_waveforms, inflow_time, inflow_rate,
            time_index=t_idx,
        )
        snapshot.time_index = t_idx
        snapshot.time_norm = t_norm
        if t_idx < len(inflow_time):
            snapshot.time = float(inflow_time[t_idx])

        dataset.append(snapshot)

    return dataset


# ======================================================================
# Main pipeline
# ======================================================================
def convert_project(project_dir: Path, output_dir: Path, n_snapshots: int = 20,
                    junction_threshold: float = 2.0) -> None:
    """Convert a single SimVascular project to PyG dataset."""
    logger.info("=" * 60)
    logger.info("Processing project: %s", project_dir.name)
    logger.info("=" * 60)

    # --- 1. Parse paths ---
    logger.info("\n📍 Parsing Paths...")
    paths_dir = project_dir / "Paths"
    paths = []
    if paths_dir.exists():
        for pth_file in sorted(paths_dir.glob("*.pth")):
            path = parse_path(pth_file)
            logger.info("  %s: %d points", path.name, len(path.points))
            paths.append(path)

    if not paths:
        logger.error("❌ No paths found in %s", paths_dir)
        return

    # --- 2. Parse segmentations ---
    logger.info("\n📐 Parsing Segmentations...")
    seg_dir = project_dir / "Segmentations"
    contour_groups = []
    if seg_dir.exists():
        for ctgr_file in sorted(seg_dir.glob("*.ctgr")):
            cg = parse_contour_group(ctgr_file)
            logger.info("  %s: %d contours, radius=[%.3f, %.3f]",
                        cg.path_name, len(cg.radii),
                        cg.radii.min() if len(cg.radii) > 0 else 0,
                        cg.radii.max() if len(cg.radii) > 0 else 0)
            contour_groups.append(cg)

    # --- 3. Parse pressure waveforms ---
    logger.info("\n📈 Parsing Pressure Waveforms (COR_Pim)...")
    sim_dir = project_dir / "Simulations"
    pressure_waveforms = []
    if sim_dir.exists():
        for pim_file in sorted(sim_dir.rglob("COR_Pim_*")):
            pw = parse_pressure_waveform(pim_file)
            logger.info("  %s: %d steps, P=[%.0f, %.0f]",
                        pw.name, len(pw.pressure),
                        pw.pressure.min(), pw.pressure.max())
            pressure_waveforms.append(pw)

    # --- 4. Parse inflow ---
    logger.info("\n🌊 Parsing Inflow...")
    inflow_time = np.array([], dtype=np.float32)
    inflow_rate = np.array([], dtype=np.float32)
    flow_dir = project_dir / "flow-files"
    if flow_dir.exists():
        inflow_file = flow_dir / "inflow_1d.flow"
        if inflow_file.exists():
            inflow_time, inflow_rate = parse_inflow(inflow_file)
            logger.info("  inflow_1d.flow: %d points, Q=[%.1f, %.1f]",
                        len(inflow_rate), inflow_rate.min(), inflow_rate.max())

    # --- 5. Build graph + labels ---
    logger.info("\n🔨 Building Graph...")
    dataset = build_temporal_dataset(
        paths, contour_groups, pressure_waveforms,
        inflow_time, inflow_rate,
        n_snapshots=n_snapshots,
        junction_threshold=junction_threshold,
    )

    # --- 6. Save --- each project gets its own subdirectory
    project_name = project_dir.name
    project_out  = output_dir / project_name   # data/processed/<project_name>/
    project_out.mkdir(parents=True, exist_ok=True)

    for i, data in enumerate(dataset):
        out_path = project_out / f"{project_name}_t{i:03d}.pt"
        torch.save(data, out_path)

    combined_path = project_out / f"{project_name}_all.pt"
    torch.save(dataset, combined_path)

    # --- 7. Summary ---
    sample = dataset[0]
    logger.info("\n" + "=" * 60)
    logger.info("✅ CONVERSION COMPLETE")
    logger.info("=" * 60)
    logger.info("  Project:     %s", project_name)
    logger.info("  Snapshots:   %d", len(dataset))
    logger.info("  Nodes:       %d", sample.num_nodes)
    logger.info("  Edges:       %d", sample.edge_index.shape[1])
    logger.info("  x shape:     %s  (x, y, z, radius, is_junction, time_norm)", list(sample.x.shape))
    logger.info("  edge_attr:   %s  (avg_radius, length)", list(sample.edge_attr.shape))
    logger.info("  edge_index:  %s", list(sample.edge_index.shape))
    logger.info("  y shape:     %s  (pressure, velocity)", list(sample.y.shape))
    logger.info("  Output dir:  %s", project_out)
    logger.info("")
    logger.info("  📦 Files saved:")
    for f in sorted(project_out.glob(f"{project_name}*")):
        size_kb = f.stat().st_size / 1024
        logger.info("     %s (%.1f KB)", f.name, size_kb)

    # Save manifest
    manifest = {
        "project": project_name,
        "n_snapshots": len(dataset),
        "n_nodes": int(sample.num_nodes),
        "n_edges": int(sample.edge_index.shape[1]),
        "x_shape": list(sample.x.shape),
        "edge_attr_shape": list(sample.edge_attr.shape),
        "y_shape": list(sample.y.shape),
        "paths": {p.name: len(p.points) for p in paths},
        "feature_names": {
            "x": ["coord_x", "coord_y", "coord_z", "radius", "is_junction", "time_norm"],
            "edge_attr": ["avg_radius", "segment_length"],
            "y": ["pressure_dyne_cm2", "velocity_cm_s"],
        },
    }
    manifest_path = project_out / f"{project_name}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("  📋 Manifest → %s", manifest_path.name)


# ======================================================================
# Validation
# ======================================================================
def validate_dataset(output_dir: Path, project_name: str) -> None:
    """Validate the generated dataset for MeshGraphNet compatibility."""
    # Support both old flat layout and new per-project subdirectory layout
    subdir_path = output_dir / project_name / f"{project_name}_all.pt"
    flat_path   = output_dir / f"{project_name}_all.pt"
    combined_path = subdir_path if subdir_path.exists() else flat_path
    if not combined_path.exists():
        logger.error("❌ File not found: %s", combined_path)
        return

    dataset = torch.load(combined_path, weights_only=False)
    logger.info("\n🔍 Validating %d graphs...", len(dataset))

    for i, data in enumerate(dataset):
        # Shape checks
        assert data.x.dim() == 2 and data.x.shape[1] == 6, \
            f"Graph {i}: x shape {data.x.shape}, expected [N, 6]"
        assert data.edge_index.dim() == 2 and data.edge_index.shape[0] == 2, \
            f"Graph {i}: edge_index shape {data.edge_index.shape}"
        assert data.edge_attr.dim() == 2 and data.edge_attr.shape[1] == 2, \
            f"Graph {i}: edge_attr shape {data.edge_attr.shape}, expected [E, 2]"
        assert data.y.dim() == 2 and data.y.shape[1] == 2, \
            f"Graph {i}: y shape {data.y.shape}, expected [N, 2]"

        # Consistency checks
        assert data.edge_index.max() < data.x.shape[0], \
            f"Graph {i}: edge references non-existent node"
        assert data.edge_attr.shape[0] == data.edge_index.shape[1], \
            f"Graph {i}: edge_attr/edge_index size mismatch"

        # Dtype checks (MeshGraphNet compatibility)
        assert data.x.dtype == torch.float32, f"Graph {i}: x dtype {data.x.dtype}"
        assert data.edge_attr.dtype == torch.float32
        assert data.edge_index.dtype == torch.long
        assert data.y.dtype == torch.float32

        # Value checks
        assert not torch.isnan(data.x).any(), f"Graph {i}: NaN in x"
        assert not torch.isnan(data.edge_attr).any(), f"Graph {i}: NaN in edge_attr"
        assert not torch.isnan(data.y).any(), f"Graph {i}: NaN in y"
        assert (data.edge_attr[:, 1] >= 0).all(), f"Graph {i}: negative edge lengths"

    # MeshGraphNet forward pass simulation
    sample = dataset[0]
    node_features = sample.x          # [N, 5]
    edge_features = sample.edge_attr  # [E, 2]
    edge_idx = sample.edge_index      # [2, E]
    logger.info("  MeshGraphNet interface check:")
    logger.info("    node_features: %s %s", list(node_features.shape), node_features.dtype)
    logger.info("    edge_features: %s %s", list(edge_features.shape), edge_features.dtype)
    logger.info("    edge_index:    %s %s", list(edge_idx.shape), edge_idx.dtype)

    logger.info("✅ All %d graphs passed validation!", len(dataset))


# ======================================================================
# Batch conversion
# ======================================================================
def is_already_converted(output_dir: Path, project_name: str) -> bool:
    """Return True if the project's manifest already exists in the output subdir."""
    manifest = output_dir / project_name / f"{project_name}_manifest.json"
    return manifest.exists()


def convert_batch(
    raw_dir:    Path,
    output_dir: Path,
    n_snapshots: int  = 20,
    junction_threshold: float = 2.0,
    filter_str: str   = "",
    skip_existing: bool = True,
    validate:   bool  = False,
) -> None:
    """
    Convert every SimVascular project found in raw_dir.

    raw_dir layout expected:
        raw_dir/
          0074_H_CORO_KD/   ← project folder (must contain Paths/ subdir)
          0070_H_CORO_KD/
          ...

    Output layout:
        output_dir/
          0074_H_CORO_KD/   ← one subdir per project
            0074_H_CORO_KD_all.pt
            0074_H_CORO_KD_manifest.json
            ...
          0070_H_CORO_KD/
            ...
    """
    if not raw_dir.exists():
        logger.error("❌ raw_dir not found: %s", raw_dir)
        return

    # Collect candidate project directories (must contain Paths/ subfolder)
    candidates = sorted(
        d for d in raw_dir.iterdir()
        if d.is_dir() and (d / "Paths").exists()
    )

    # Optional name filter (e.g. "CORO_KD" to only process coronary KD cases)
    if filter_str:
        candidates = [d for d in candidates if filter_str.upper() in d.name.upper()]

    if not candidates:
        logger.warning("No valid SimVascular project folders found in %s", raw_dir)
        return

    logger.info("Found %d project(s) to process (filter=%r):", len(candidates),
                filter_str or "<none>")
    skipped = []
    to_run  = []
    for d in candidates:
        if skip_existing and is_already_converted(output_dir, d.name):
            skipped.append(d.name)
        else:
            to_run.append(d)

    for name in skipped:
        logger.info("  ⏭  SKIP  %s  (already converted)", name)
    for d in to_run:
        logger.info("  ▶  QUEUE %s", d.name)

    if not to_run:
        logger.info("Nothing to do — all projects already converted.")
        return

    results = {"ok": [], "failed": []}
    for i, project_dir in enumerate(to_run, 1):
        logger.info("\n[%d/%d] ─────────────────────────────────", i, len(to_run))
        try:
            convert_project(project_dir, output_dir,
                            n_snapshots=n_snapshots,
                            junction_threshold=junction_threshold)
            if validate:
                validate_dataset(output_dir, project_dir.name)
            results["ok"].append(project_dir.name)
        except Exception as exc:
            logger.error("❌ FAILED %s: %s", project_dir.name, exc, exc_info=True)
            results["failed"].append(project_dir.name)

    logger.info("\n" + "=" * 60)
    logger.info("BATCH COMPLETE  ✅ %d succeeded  ❌ %d failed",
                len(results["ok"]), len(results["failed"]))
    if results["failed"]:
        logger.info("  Failed: %s", results["failed"])
    logger.info("=" * 60)


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Convert SimVascular project(s) to PyG dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output layout
─────────────
  data/processed/
    0074_H_CORO_KD/
      0074_H_CORO_KD_all.pt
      0074_H_CORO_KD_manifest.json
      0074_H_CORO_KD_t000.pt  ...
    0070_H_CORO_KD/
      ...

Examples
────────
  # Convert one project
  python convert_all.py --project /projects/0074_H_CORO_KD --output ./data/processed

  # Batch-convert all projects in raw_dir (skip already done)
  python convert_all.py --batch --raw_dir ./data/raw --output ./data/processed

  # Batch, only coronary KD cases, validate each
  python convert_all.py --batch --raw_dir ./data/raw --output ./data/processed \\
    --filter CORO_KD --validate

  # Re-convert even if already done
  python convert_all.py --batch --raw_dir ./data/raw --output ./data/processed \\
    --no_skip

  # Validate a single project
  python convert_all.py --validate_only --output ./data/processed --name 0074_H_CORO_KD
        """,
    )
    # --- Single project ---
    parser.add_argument(
        "--project", type=Path,
        default=None,
        help="Single SimVascular project directory to convert",
    )
    # --- Batch mode ---
    parser.add_argument(
        "--batch", action="store_true",
        help="Convert all projects found in --raw_dir",
    )
    parser.add_argument(
        "--raw_dir", type=Path,
        default=Path("./data/raw"),
        help="Root folder containing raw SimVascular projects (batch mode)",
    )
    parser.add_argument(
        "--filter", type=str, default="",
        metavar="STR",
        help="Only process projects whose name contains STR (case-insensitive, batch mode)",
    )
    parser.add_argument(
        "--no_skip", action="store_true",
        help="Re-convert even if output already exists (batch mode)",
    )
    # --- Common ---
    parser.add_argument(
        "--output", type=Path,
        default=Path("./data/processed"),
        help="Root output directory (each project gets a subdirectory)",
    )
    parser.add_argument(
        "-n", "--n_snapshots", type=int, default=20,
        help="Number of temporal snapshots per project (default: 20)",
    )
    parser.add_argument(
        "--junction_threshold", type=float, default=2.0,
        help="Max distance (cm) to detect path junctions (default: 2.0)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run validation after conversion",
    )
    parser.add_argument(
        "--validate_only", action="store_true",
        help="Skip conversion, only validate an existing project",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Project name for --validate_only mode",
    )
    args = parser.parse_args()

    # --- Validate only ---
    if args.validate_only:
        name = args.name or (args.project.name if args.project else None)
        if not name:
            parser.error("--validate_only requires --name or --project")
        validate_dataset(args.output, name)
        return

    # --- Batch mode ---
    if args.batch:
        convert_batch(
            raw_dir    = args.raw_dir,
            output_dir = args.output,
            n_snapshots = args.n_snapshots,
            junction_threshold = args.junction_threshold,
            filter_str = args.filter,
            skip_existing = not args.no_skip,
            validate   = args.validate,
        )
        return

    # --- Single project ---
    if args.project is None:
        parser.error("Provide --project <dir> or use --batch")
    convert_project(args.project, args.output,
                    n_snapshots=args.n_snapshots,
                    junction_threshold=args.junction_threshold)
    if args.validate:
        validate_dataset(args.output, args.project.name)


if __name__ == "__main__":
    main()
