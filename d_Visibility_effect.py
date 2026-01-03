# -*- coding: utf-8 -*-
"""
Visibility filtering on CAD meshes (Open3D) - Synthetic dataset generator

What this script does:
- Load bolt + washer triangle meshes (.ply)
- Apply translations + rotation (loosening angle)
- Apply a "view direction" (tilt) by rotating the whole scene
- Detect "shadowed" vertices using a top-triangle test in partitions
- Export remaining (visible) vertices to TXT

Typical usage:
python scripts/visibility_effect.py \
  --bolt-ply data/cad/Bolt_v15_remesh_cI1.ply \
  --washer-ply data/cad/Washer_v1_remesh_cI1.ply \
  --out-root outputs/synthetic_visibility \
  --length-start 0 --length-end 1 \
  --angle-step 10 \
  --partitions 15

Repo notes:
- Input CAD meshes should live in data/ (gitignored if large)
- Outputs should go to outputs/ (gitignored)
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import open3d as o3d


log = logging.getLogger("visibility_effect")


# -----------------------------
# Geometry helpers
# -----------------------------
def rodrigues_rotation(mesh: o3d.geometry.TriangleMesh, direction, reverse: int) -> o3d.geometry.TriangleMesh:
    """
    Rotate mesh to align direction vector with +Z using Rodrigues' formula.
    reverse = +1 applies rotation, reverse = -1 un-rotates.
    """
    n1 = np.array(direction, dtype=float)
    n2 = np.array([0.0, 0.0, 1.0], dtype=float)

    # Special case: n1 ~ (0,0,-1)
    if np.allclose(n1, [0, 0, -1], atol=1e-3):
        theta = np.pi
        rotation_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        n1 = n1 / (np.linalg.norm(n1) + 1e-12)
        n2 = n2 / (np.linalg.norm(n2) + 1e-12)

        rotation_axis = np.cross(n1, n2)
        axis_norm = np.linalg.norm(rotation_axis)
        if axis_norm < 1e-12:
            # Already aligned
            return mesh
        rotation_axis = rotation_axis / axis_norm

        cos_theta = np.clip(np.dot(n1, n2), -1.0, 1.0)
        theta = float(np.arccos(cos_theta))

    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ], dtype=float)

    R = np.eye(3) + reverse * np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return mesh.rotate(R, center=(0, 0, 0))


def filter_mesh_by_z(mesh: o3d.geometry.TriangleMesh, z_threshold: float) -> o3d.geometry.TriangleMesh:
    """
    Keep vertices with z < z_threshold and triangles fully inside.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    valid_vertex_indices = np.where(vertices[:, 2] < z_threshold)[0]
    valid_set = set(valid_vertex_indices.tolist())

    valid_triangles = [tri for tri in triangles if all(v in valid_set for v in tri)]
    index_map = {old: new for new, old in enumerate(valid_vertex_indices)}
    remapped_triangles = np.array([[index_map[v] for v in tri] for tri in valid_triangles], dtype=np.int32)

    filtered_vertices = vertices[valid_vertex_indices]

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(filtered_vertices)
    out.triangles = o3d.utility.Vector3iVector(remapped_triangles)
    return out


def build_vertex_triangle_map(triangles: np.ndarray) -> dict[int, list[int]]:
    vt = {}
    for tri_idx, tri in enumerate(triangles):
        for v in tri:
            vt.setdefault(int(v), []).append(tri_idx)
    return vt


@dataclass
class TriangleData:
    tri_idx: int
    v1_idx: int
    v2_idx: int
    v3_idx: int
    v1: np.ndarray
    v2: np.ndarray
    v3: np.ndarray
    normal: np.ndarray
    d: float
    edge1: np.ndarray
    edge2: np.ndarray
    edge3: np.ndarray

    @staticmethod
    def from_triangle(tri_indices: np.ndarray, vertices: np.ndarray, tri_idx: int) -> "TriangleData":
        v1i, v2i, v3i = map(int, tri_indices)
        v1, v2, v3 = vertices[tri_indices]
        normal = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(normal) + 1e-12
        normal = normal / norm
        a, b, c = normal
        d = -(a * v1[0] + b * v1[1] + c * v1[2])
        return TriangleData(
            tri_idx=tri_idx,
            v1_idx=v1i, v2_idx=v2i, v3_idx=v3i,
            v1=v1, v2=v2, v3=v3,
            normal=normal,
            d=float(d),
            edge1=(v2 - v1),
            edge2=(v3 - v2),
            edge3=(v1 - v3),
        )


def detect_shadowed_points_top_triangles(mesh: o3d.geometry.TriangleMesh) -> set[int]:
    """
    Shadow test along +Z (after the scene has been rotated so the viewing direction becomes +Z).
    Returns a set of vertex indices that are shadowed.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    vt_map = build_vertex_triangle_map(triangles)
    z_sorted = np.argsort(-vertices[:, 2])

    shadowed = set()
    considering = []
    top_tris: list[TriangleData] = []

    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)

    for idx in z_sorted:
        p = vertices[idx]
        is_shadowed = False

        for tri in top_tris:
            if idx in (tri.v1_idx, tri.v2_idx, tri.v3_idx):
                continue

            a, b, c = tri.normal
            if abs(c) < 1e-12:
                continue

            z_proj = -(a * p[0] + b * p[1] + tri.d) / c
            proj = np.array([p[0], p[1], z_proj], dtype=float)

            c1 = np.cross(tri.edge1, proj - tri.v1)
            c2 = np.cross(tri.edge2, proj - tri.v2)
            c3 = np.cross(tri.edge3, proj - tri.v3)

            if all(np.dot(cc, tri.normal) >= 0 for cc in (c1, c2, c3)):
                # behind along +Z
                if np.dot(p - tri.v1, z_axis) < 0:
                    is_shadowed = True
                    break

        if is_shadowed:
            shadowed.add(int(idx))
        else:
            considering.append(int(idx))
            for tri_idx in vt_map.get(int(idx), []):
                tri_vertices = triangles[tri_idx]
                if all(int(v) in considering for v in tri_vertices):
                    top_tris.append(TriangleData.from_triangle(tri_vertices, vertices, tri_idx))

    # recheck
    rechecked = set(considering)
    for idx in list(rechecked):
        p = vertices[idx]
        is_shadowed = False

        for tri in top_tris:
            if idx in (tri.v1_idx, tri.v2_idx, tri.v3_idx):
                continue

            a, b, c = tri.normal
            if abs(c) < 1e-12:
                continue

            z_proj = -(a * p[0] + b * p[1] + tri.d) / c
            proj = np.array([p[0], p[1], z_proj], dtype=float)

            c1 = np.cross(tri.edge1, proj - tri.v1)
            c2 = np.cross(tri.edge2, proj - tri.v2)
            c3 = np.cross(tri.edge3, proj - tri.v3)

            if all(np.dot(cc, tri.normal) >= 0 for cc in (c1, c2, c3)):
                if np.dot(p - tri.v1, z_axis) < 0:
                    is_shadowed = True
                    break

        if is_shadowed:
            shadowed.add(int(idx))
            if idx in considering:
                considering.remove(idx)

    return shadowed


def divide_into_partitions(mesh: o3d.geometry.TriangleMesh, px: int, py: int, overlap: float):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vt_map = build_vertex_triangle_map(triangles)

    min_x, min_y = np.min(vertices[:, :2], axis=0)
    max_x, max_y = np.max(vertices[:, :2], axis=0)
    w = (max_x - min_x) / px
    h = (max_y - min_y) / py
    ox = w * overlap
    oy = h * overlap

    partitions = []
    for i in range(px):
        for j in range(py):
            x_min = min_x + i * w - ox
            x_max = min_x + (i + 1) * w + ox
            y_min = min_y + j * h - oy
            y_max = min_y + (j + 1) * h + oy

            vset = set(np.where(
                (vertices[:, 0] >= x_min) & (vertices[:, 0] <= x_max) &
                (vertices[:, 1] >= y_min) & (vertices[:, 1] <= y_max)
            )[0].tolist())

            tris_in = set()
            for v_idx in vset:
                tris_in.update(vt_map.get(int(v_idx), []))

            for tri_idx in tris_in:
                vset.update(triangles[tri_idx].tolist())

            v_idx_sorted = np.array(sorted(vset), dtype=int)
            idx_map = {old: new for new, old in enumerate(v_idx_sorted)}

            valid_tri_idxs = [ti for ti in tris_in if all(v in vset for v in triangles[ti])]
            part_vertices = vertices[v_idx_sorted]
            part_triangles = triangles[valid_tri_idxs]
            part_triangles = np.array([[idx_map[int(v)] for v in tri] for tri in part_triangles], dtype=np.int32)

            pm = o3d.geometry.TriangleMesh()
            pm.vertices = o3d.utility.Vector3dVector(part_vertices)
            pm.triangles = o3d.utility.Vector3iVector(part_triangles)

            partitions.append({"mesh": pm, "index_map": idx_map, "i": i, "j": j})

    return partitions


def filter_shadowed_in_nonoverlap_region(full_mesh: o3d.geometry.TriangleMesh,
                                        part_mesh: o3d.geometry.TriangleMesh,
                                        part_index: int,
                                        shadowed_points: set[int],
                                        partitions_x: int,
                                        partitions_y: int):
    """
    Keep only shadowed points that fall within the non-overlap core region of this partition.
    This reduces double counting across overlapped partitions.
    """
    vertices = np.asarray(full_mesh.vertices)
    min_x, min_y = np.min(vertices[:, :2], axis=0)
    max_x, max_y = np.max(vertices[:, :2], axis=0)
    w = (max_x - min_x) / partitions_x
    h = (max_y - min_y) / partitions_y

    # Determine partition coords from index
    ix = part_index // partitions_x
    iy = part_index % partitions_x

    x_min = min_x + ix * w
    x_max = min_x + (ix + 1) * w
    y_min = min_y + iy * h
    y_max = min_y + (iy + 1) * h

    v0 = np.asarray(part_mesh.vertices)
    filtered = {idx for idx in shadowed_points
                if x_min <= v0[idx][0] <= x_max and y_min <= v0[idx][1] <= y_max}
    return filtered


def export_remaining_points(mesh: o3d.geometry.TriangleMesh,
                            shadowed_points: set[int],
                            out_txt: Path):
    vertices = np.asarray(mesh.vertices)
    remaining = np.array([vertices[i] for i in range(len(vertices)) if i not in shadowed_points], dtype=float)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_txt, remaining, fmt="%.6f", comments="")


# -----------------------------
# Simulation / generation
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Visibility filtering (shadow removal) for CAD bolt+washer meshes.")

    p.add_argument("--bolt-ply", type=str, required=True, help="Bolt mesh path (.ply).")
    p.add_argument("--washer-ply", type=str, required=True, help="Washer mesh path (.ply).")

    p.add_argument("--out-root", type=str, default="outputs/visibility_effect",
                   help="Output root folder.")
    p.add_argument("--dataset-name", type=str, default="bolt_length_dataset",
                   help="Subfolder name inside out-root.")

    # Loosening length loop
    p.add_argument("--length-start", type=int, default=0, help="Start length index (inclusive).")
    p.add_argument("--length-end", type=int, default=1, help="End length index (exclusive).")
    p.add_argument("--length-step", type=int, default=1, help="Step length index.")

    # Angle loop
    p.add_argument("--angle-start", type=int, default=0, help="Start angle (deg).")
    p.add_argument("--angle-end", type=int, default=60, help="End angle (deg, exclusive).")
    p.add_argument("--angle-step", type=int, default=10, help="Angle step (deg).")

    # Tilt / view direction
    p.add_argument("--tilt-ax-deg", type=float, default=15.0, help="Tilt ax angle (deg).")
    p.add_argument("--tilt-by-deg", type=float, default=0.0, help="Tilt by angle (deg).")

    # Partitioning
    p.add_argument("--partitions", type=int, default=15, help="Number of partitions per axis (px=py).")
    p.add_argument("--overlap", type=float, default=0.1, help="Partition overlap fraction.")

    # Z filtering for bolt mesh
    p.add_argument("--bolt-head-thickness", type=float, default=13.0,
                   help="Bolt head thickness used in z-threshold formula.")
    p.add_argument("--washer-thickness", type=float, default=2.0,
                   help="Washer thickness for z shift.")
    p.add_argument("--z-buffer", type=float, default=1.0,
                   help="Extra margin in z threshold.")

    # Single translation or multiple
    p.add_argument("--single", action="store_true",
                   help="Use only one bolt/washer at origin (default).")

    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    bolt_path = Path(args.bolt_ply)
    washer_path = Path(args.washer_ply)
    if not bolt_path.exists() or not washer_path.exists():
        raise FileNotFoundError("Bolt or washer ply not found.")

    bolt = o3d.io.read_triangle_mesh(str(bolt_path))
    washer = o3d.io.read_triangle_mesh(str(washer_path))
    if not bolt.has_vertices() or not washer.has_vertices():
        raise ValueError("Loaded mesh has no vertices.")

    # Translations: keep your original grid available; default to single
    if args.single:
        translations = np.array([[0.0, 0.0, 0.0]], dtype=float)
    else:
        translations = np.array([
            [-140,  -60, 0],
            [ -40,  -60, 0],
            [  40,  -60, 0],
            [ 140,  -60, 0],
            [-140,   60, 0],
            [ -40,   60, 0],
            [  40,   60, 0],
            [ 140,   60, 0],
        ], dtype=float)

    # View direction params (your original: ax=-tan(tilt), by=-tan(tilt))
    ax = -np.tan(np.deg2rad(np.array([args.tilt_ax_deg], dtype=float)))
    by = -np.tan(np.deg2rad(np.array([args.tilt_by_deg], dtype=float)))

    out_root = Path(args.out_root) / args.dataset_name

    for length_idx in range(args.length_start, args.length_end, args.length_step):
        count = 0

        # Create folder per length (same spirit as your original)
        length_dir = out_root / f"bolt_len_{length_idx}" / "test"
        length_dir.mkdir(parents=True, exist_ok=True)

        for angle_deg in range(args.angle_start, args.angle_end, args.angle_step):
            for a in ax:
                for b in by:
                    count += 1
                    log.info("length=%d | angle=%d | count=%d", length_idx, angle_deg, count)

                    # Build scene meshes
                    all_meshes = []

                    # Bolt translation / z shift rule (your original)
                    for t in translations:
                        t_copy = np.array(t, dtype=float)
                        t_copy[2] = t_copy[2] - args.washer_thickness - args.bolt_head_thickness - length_idx / 5.0

                        bolt_instance = copy.deepcopy(bolt)

                        z_thresh = args.bolt_head_thickness + args.z_buffer + max(length_idx, 5) / 5.0
                        bolt_instance = filter_mesh_by_z(bolt_instance, z_threshold=z_thresh)

                        # Rotate around Z axis
                        rot_axis = np.array([0.0, 0.0, 1.0], dtype=float)
                        Rz = o3d.geometry.get_rotation_matrix_from_axis_angle(np.deg2rad(angle_deg) * rot_axis)
                        bolt_instance.rotate(Rz, center=(0, 0, 0))

                        bolt_instance.translate(t_copy, relative=True)
                        all_meshes.append(bolt_instance)

                    for t in translations:
                        t_copy = np.array(t, dtype=float)
                        t_copy[2] = t_copy[2] - args.washer_thickness
                        washer_instance = copy.deepcopy(washer)
                        washer_instance.translate(t_copy, relative=True)
                        all_meshes.append(washer_instance)

                    # Combine into one mesh
                    combined = o3d.geometry.TriangleMesh()
                    for m in all_meshes:
                        combined += m

                    # Define view direction (your original: direction = (-a,-b,-1))
                    direction = (-float(a), -float(b), -1.0)

                    # Rotate scene to align view direction with +Z
                    combined = rodrigues_rotation(combined, direction, reverse=1)

                    # Partition + detect
                    px = py = int(args.partitions)
                    parts = divide_into_partitions(combined, partitions_x=px, partitions_y=py, overlap=float(args.overlap))

                    combined_shadowed = set()

                    for p_idx, part in enumerate(parts):
                        part_mesh = part["mesh"]
                        idx_map = part["index_map"]
                        rev_map = {v: k for k, v in idx_map.items()}

                        shadowed_local = detect_shadowed_points_top_triangles(part_mesh)
                        shadowed_local = filter_shadowed_in_nonoverlap_region(
                            combined, part_mesh, p_idx, shadowed_local, partitions_x=px, partitions_y=py
                        )

                        # Map local indices back to combined mesh indices
                        shadowed_global = {rev_map[li] for li in shadowed_local if li in rev_map}
                        combined_shadowed.update(shadowed_global)

                    # Un-rotate back to original orientation
                    combined = rodrigues_rotation(combined, direction, reverse=-1)

                    out_txt = length_dir / f"Boltlen_{count}.txt"
                    export_remaining_points(combined, combined_shadowed, out_txt)

    log.info("Done. Output written under: %s", out_root.as_posix())


if __name__ == "__main__":
    main()
