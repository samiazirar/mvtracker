#!/usr/bin/env python3
"""Generate static point-cloud templates for external grippers."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import kinpy as kp
import numpy as np
import open3d as o3d
import trimesh
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return a 3x3 rotation matrix for the given roll-pitch-yaw (XYZ extrinsic)."""
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def _origin_matrix(xyz: Tuple[float, float, float], rpy: Tuple[float, float, float]) -> np.ndarray:
    """Compose a 4x4 homogeneous transform from xyz / rpy."""
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = _rpy_to_matrix(*rpy)
    mat[:3, 3] = xyz
    return mat


def _resolve_package_path(filename: str, package_map: Dict[str, Path], urdf_dir: Path) -> Path:
    """Resolve mesh filename that may use package:// syntax."""
    if filename.startswith("package://"):
        pkg_and_path = filename[len("package://") :]
        pkg, rel = pkg_and_path.split("/", 1)
        base = package_map.get(pkg)
        if base is None:
            raise FileNotFoundError(f"Package '{pkg}' not provided in package_map")
        return base / rel
    path = Path(filename)
    if not path.is_absolute():
        return urdf_dir / path
    return path


def _apply_scale(vertices: np.ndarray, scale: Tuple[float, float, float]) -> np.ndarray:
    sx, sy, sz = scale
    return vertices * np.array([sx, sy, sz], dtype=np.float64)


def _clone_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    clone = o3d.geometry.TriangleMesh()
    clone.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    clone.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles, dtype=np.int32))
    if mesh.has_vertex_normals():
        clone.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals, dtype=np.float64))
    if mesh.has_vertex_colors():
        clone.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors, dtype=np.float64))
    return clone


@dataclass
class VisualElement:
    link: str
    geom_type: str
    origin: np.ndarray  # 4x4 matrix
    filename: Optional[str] = None
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    size: Optional[Tuple[float, float, float]] = None
    radius: Optional[float] = None
    length: Optional[float] = None
    color: Tuple[float, float, float] = (0.7, 0.7, 0.7)


class URDFGripperModel:
    def __init__(
        self,
        urdf_path: Path,
        root_link: str,
        package_map: Dict[str, Path],
        default_color: Tuple[float, float, float] = (0.6, 0.6, 0.6),
    ) -> None:
        self.urdf_path = urdf_path
        self.package_map = package_map
        self.root_link = root_link
        self.default_color = default_color

        urdf_text = urdf_path.read_text()
        self.chain = kp.build_chain_from_urdf(urdf_text.encode("utf-8"))
        self.joint_names = self.chain.get_joint_parameter_names()
        self.visuals = self._parse_visuals(urdf_text)
        self._mesh_cache: Dict[str, o3d.geometry.TriangleMesh] = {}

    def _parse_visuals(self, urdf_text: str) -> List[VisualElement]:
        visuals: List[VisualElement] = []
        tree = ET.fromstring(urdf_text)
        for link_el in tree.findall("link"):
            link_name = link_el.get("name")
            if not link_name:
                continue
            for visual_el in link_el.findall("visual"):
                origin_el = visual_el.find("origin")
                if origin_el is not None:
                    xyz = tuple(float(v) for v in origin_el.get("xyz", "0 0 0").split())
                    rpy = tuple(float(v) for v in origin_el.get("rpy", "0 0 0").split())
                else:
                    xyz = (0.0, 0.0, 0.0)
                    rpy = (0.0, 0.0, 0.0)
                origin = _origin_matrix(xyz, rpy)

                color = self.default_color
                material_el = visual_el.find("material")
                if material_el is not None:
                    color_el = material_el.find("color")
                    if color_el is not None and "rgba" in color_el.attrib:
                        rgba = [float(v) for v in color_el.get("rgba").split()]
                        if len(rgba) >= 3:
                            color = tuple(rgba[:3])

                geom_el = visual_el.find("geometry")
                if geom_el is None:
                    continue

                mesh_el = geom_el.find("mesh")
                if mesh_el is not None and "filename" in mesh_el.attrib:
                    scale = tuple(float(v) for v in mesh_el.get("scale", "1 1 1").split())
                    visuals.append(
                        VisualElement(
                            link=link_name,
                            geom_type="mesh",
                            origin=origin,
                            filename=mesh_el.get("filename"),
                            scale=scale,
                            color=color,
                        )
                    )
                    continue

                box_el = geom_el.find("box")
                if box_el is not None and "size" in box_el.attrib:
                    size = tuple(float(v) for v in box_el.get("size").split())
                    visuals.append(
                        VisualElement(
                            link=link_name,
                            geom_type="box",
                            origin=origin,
                            size=size,
                            color=color,
                        )
                    )
                    continue

                cyl_el = geom_el.find("cylinder")
                if cyl_el is not None and {"radius", "length"} <= cyl_el.attrib.keys():
                    visuals.append(
                        VisualElement(
                            link=link_name,
                            geom_type="cylinder",
                            origin=origin,
                            radius=float(cyl_el.get("radius")),
                            length=float(cyl_el.get("length")),
                            color=color,
                        )
                    )
                    continue
        return visuals

    def _load_mesh(self, filename: str, scale: Tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
        cache_key = f"{filename}|{scale}"
        if cache_key in self._mesh_cache:
            return self._mesh_cache[cache_key]
        mesh_path = _resolve_package_path(filename, self.package_map, self.urdf_path.parent)
        suffix = mesh_path.suffix.lower()
        if suffix == ".dae":
            tm = trimesh.load_mesh(str(mesh_path), process=False)
            if isinstance(tm, trimesh.Scene):
                tm = trimesh.util.concatenate(tuple(tm.geometry.values()))
            if tm.is_empty:
                raise FileNotFoundError(f"Failed to load mesh: {mesh_path}")
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(np.asarray(tm.vertices, dtype=np.float64))
            mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces, dtype=np.int32))
        else:
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            if mesh.is_empty():
                raise FileNotFoundError(f"Failed to load mesh: {mesh_path}")
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        vertices = _apply_scale(vertices, scale)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.compute_vertex_normals()
        self._mesh_cache[cache_key] = mesh
        return mesh

    def sample_point_cloud(
        self,
        joint_values: Optional[Dict[str, float]] = None,
        samples_per_visual: int = 1500,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if joint_values is None:
            joint_values = {name: 0.0 for name in self.joint_names}
        else:
            joint_values = {name: joint_values.get(name, 0.0) for name in self.joint_names}

        fk = self.chain.forward_kinematics(joint_values)
        if self.root_link not in fk:
            raise KeyError(f"Root link '{self.root_link}' not present in kinematics map")
        root_inv = np.linalg.inv(fk[self.root_link].matrix())

        all_points: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []

        for vis in self.visuals:
            if vis.link not in fk:
                continue
            local_T = root_inv @ fk[vis.link].matrix() @ vis.origin

            if vis.geom_type == "mesh" and vis.filename:
                mesh = self._load_mesh(vis.filename, vis.scale)
                mesh_copy = _clone_mesh(mesh)
                mesh_copy.transform(local_T)
                pcd = mesh_copy.sample_points_uniformly(number_of_points=samples_per_visual)
                pts = np.asarray(pcd.points)
                if pts.size == 0:
                    continue
                colors = (
                    np.asarray(pcd.colors)
                    if pcd.has_colors()
                    else np.ones_like(pts) * np.array(vis.color, dtype=np.float64)
                )
            elif vis.geom_type == "box" and vis.size:
                mesh = o3d.geometry.TriangleMesh.create_box(*vis.size)
                mesh.compute_vertex_normals()
                mesh.transform(local_T)
                pcd = mesh.sample_points_uniformly(number_of_points=samples_per_visual)
                pts = np.asarray(pcd.points)
                if pts.size == 0:
                    continue
                colors = np.ones_like(pts) * np.array(vis.color, dtype=np.float64)
            elif vis.geom_type == "cylinder" and vis.radius is not None and vis.length is not None:
                mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=vis.radius, height=vis.length)
                mesh.compute_vertex_normals()
                mesh.transform(local_T)
                pcd = mesh.sample_points_uniformly(number_of_points=samples_per_visual)
                pts = np.asarray(pcd.points)
                if pts.size == 0:
                    continue
                colors = np.ones_like(pts) * np.array(vis.color, dtype=np.float64)
            else:
                continue

            all_points.append(pts)
            all_colors.append(colors)

        if not all_points:
            return np.empty((0, 3), dtype=np.float32), None
        points = np.concatenate(all_points, axis=0).astype(np.float32)
        colors = np.concatenate(all_colors, axis=0).astype(np.float32)
        return points, colors


# ---------------------------------------------------------------------------
# Gripper specifications
# ---------------------------------------------------------------------------

ASSET_ROOT = Path("external_assets/grippers").resolve()

GRIPPER_SPECS = {
    "robotiq_2f_85": {
        "label": "Robotiq 2F-85",
        "type": "urdf",
        "urdf": ASSET_ROOT / "robotiq_2f_85_gripper_visualization/urdf/robotiq_2f_85_model.urdf",
        "root_link": "robotiq_arg2f_base_link",
        "package_map": {
            "robotiq_2f_85_gripper_visualization": ASSET_ROOT / "robotiq_2f_85_gripper_visualization",
        },
        "mount_pose": np.eye(4, dtype=np.float64),
    },
    "dh_ag95": {
        "label": "Dahuan AG-95",
        "type": "urdf",
        "urdf": ASSET_ROOT
        / "dh_robotics_ag95_gripper/dh_robotics_ag95_description/urdf/dh_robotics_ag95.urdf",
        "root_link": "ee_link",
        "package_map": {
            "dh_robotics_ag95_description": ASSET_ROOT / "dh_robotics_ag95_gripper/dh_robotics_ag95_description",
        },
        "mount_pose": np.eye(4, dtype=np.float64),
    },
    "schunk_wsg50": {
        "label": "Schunk WSG-50",
        "type": "mesh",
        "mesh": ASSET_ROOT / "schunk_wsg50_model/meshes/wsg50_110.stl",
        "scale": (0.001, 0.001, 0.001),
        "root_link": "wsg50_mount",
        "default_color": (0.75, 0.75, 0.75),
        "mount_pose": np.eye(4, dtype=np.float64),
    },
}


def _sample_mesh_template(mesh_path: Path, scale: Tuple[float, float, float], color: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise FileNotFoundError(f"Failed to load mesh at {mesh_path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices = _apply_scale(vertices, scale)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=20000)
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    colors = (
        np.asarray(pcd.colors, dtype=np.float32)
        if pcd.has_colors()
        else np.ones_like(points) * np.array(color, dtype=np.float32)
    )
    return points, colors


def main() -> None:
    templates_dir = ASSET_ROOT / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Dict[str, object]] = {}

    for key, spec in GRIPPER_SPECS.items():
        label = spec["label"]
        print(f"Processing {label} ({key})")
        if spec["type"] == "urdf":
            model = URDFGripperModel(
                urdf_path=spec["urdf"],
                root_link=spec["root_link"],
                package_map=spec["package_map"],
            )
            points, colors = model.sample_point_cloud()
            joint_names = model.joint_names
        elif spec["type"] == "mesh":
            points, colors = _sample_mesh_template(
                mesh_path=spec["mesh"],
                scale=spec.get("scale", (1.0, 1.0, 1.0)),
                color=spec.get("default_color", (0.7, 0.7, 0.7)),
            )
            joint_names = []
        else:
            raise ValueError(f"Unsupported gripper spec type: {spec['type']}")

        output_path = templates_dir / f"{key}.npz"
        np.savez_compressed(
            output_path,
            points=points,
            colors=colors,
            mount_pose=spec.get("mount_pose", np.eye(4, dtype=np.float64)),
            root_link=spec.get("root_link"),
            joint_names=np.array(joint_names, dtype=object),
        )
        manifest[key] = {
            "label": label,
            "file": str(output_path.relative_to(ASSET_ROOT)),
            "root_link": spec.get("root_link"),
            "mount_pose": spec.get("mount_pose", np.eye(4)).tolist(),
            "joint_names": joint_names,
        }

    manifest_path = templates_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
