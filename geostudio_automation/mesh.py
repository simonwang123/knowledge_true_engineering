from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MeshMismatch:
    line_id: int
    xml_endpoints: tuple[tuple[float, float], tuple[float, float]] | None
    mesh_endpoints: tuple[tuple[float, float], tuple[float, float]] | None
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MeshValidation:
    mesh_id: int
    mesh_file: str
    node_count: int
    node_pair_count: int
    xml_line_count: int
    tolerance: float
    matched: bool
    mismatches: tuple[MeshMismatch, ...]

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["mismatches"] = [item.as_dict() for item in self.mismatches]
        return value


@dataclass(frozen=True, slots=True)
class PlyMesh:
    nodes: tuple[tuple[float, float, float], ...]
    node_pairs: dict[int, tuple[int, int]]


def validate_gsz_mesh(path: Path, tolerance: float = 1.0e-5) -> MeshValidation:
    with zipfile.ZipFile(path) as archive:
        root_xml = [name for name in archive.namelist() if "/" not in name.rstrip("/") and name.lower().endswith(".xml")]
        if len(root_xml) != 1:
            raise ValueError(f"GSZ must contain exactly one root XML file: {path}")
        root = ET.fromstring(archive.read(root_xml[0]))
        geometry = root.find("./Geometries/Geometry")
        if geometry is None:
            raise ValueError("GSZ XML has no geometry")
        mesh_id = int(geometry.findtext("MeshId", "0"))
        mesh_file = f"mesh_{mesh_id}.ply"
        try:
            mesh_data = archive.read(mesh_file)
        except KeyError as error:
            raise ValueError(f"current MeshId={mesh_id} but {mesh_file} is absent") from error

    points = {
        int(point.attrib["ID"]): (float(point.attrib["X"]), float(point.attrib["Y"]))
        for point in geometry.findall("./Points/Point")
    }
    lines = {
        int(line.findtext("ID", "0")): (
            int(line.findtext("PointID1", "0")),
            int(line.findtext("PointID2", "0")),
        )
        for line in geometry.findall("./Lines/Line")
    }
    mesh = read_ply_mesh(mesh_data)
    mismatches: list[MeshMismatch] = []
    for line_id, (point1, point2) in sorted(lines.items()):
        xml_endpoints = (points[point1], points[point2])
        pair = mesh.node_pairs.get(line_id)
        if pair is None:
            mismatches.append(MeshMismatch(line_id, xml_endpoints, None, "line has no node_pair record"))
            continue
        node1, node2 = pair
        if node1 < 1 or node2 < 1 or node1 > len(mesh.nodes) or node2 > len(mesh.nodes):
            mismatches.append(MeshMismatch(line_id, xml_endpoints, None, "node_pair references an absent node"))
            continue
        mesh_endpoints = (
            (mesh.nodes[node1 - 1][0], mesh.nodes[node1 - 1][1]),
            (mesh.nodes[node2 - 1][0], mesh.nodes[node2 - 1][1]),
        )
        direct = _point_close(xml_endpoints[0], mesh_endpoints[0], tolerance) and _point_close(
            xml_endpoints[1], mesh_endpoints[1], tolerance
        )
        reverse = _point_close(xml_endpoints[0], mesh_endpoints[1], tolerance) and _point_close(
            xml_endpoints[1], mesh_endpoints[0], tolerance
        )
        if not (direct or reverse):
            mismatches.append(MeshMismatch(line_id, xml_endpoints, mesh_endpoints, "mesh endpoints differ from XML"))

    extra = sorted(set(mesh.node_pairs) - set(lines))
    for line_id in extra:
        node1, node2 = mesh.node_pairs[line_id]
        endpoints = None
        if 1 <= node1 <= len(mesh.nodes) and 1 <= node2 <= len(mesh.nodes):
            endpoints = (
                (mesh.nodes[node1 - 1][0], mesh.nodes[node1 - 1][1]),
                (mesh.nodes[node2 - 1][0], mesh.nodes[node2 - 1][1]),
            )
        mismatches.append(MeshMismatch(line_id, None, endpoints, "mesh contains a line absent from XML"))

    return MeshValidation(
        mesh_id=mesh_id,
        mesh_file=mesh_file,
        node_count=len(mesh.nodes),
        node_pair_count=len(mesh.node_pairs),
        xml_line_count=len(lines),
        tolerance=tolerance,
        matched=not mismatches,
        mismatches=tuple(mismatches),
    )


def read_ply_mesh(data: bytes) -> PlyMesh:
    marker = b"end_header\n"
    header_end = data.find(marker)
    if header_end < 0:
        marker = b"end_header\r\n"
        header_end = data.find(marker)
    if header_end < 0:
        raise ValueError("PLY end_header marker is absent")
    body_offset = header_end + len(marker)
    header = data[:body_offset].decode("ascii")
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("only binary_little_endian PLY meshes are supported")
    counts = {
        name: int(count)
        for name, count in re.findall(r"^element\s+(\S+)\s+(\d+)\s*$", header, flags=re.MULTILINE)
    }
    required = {"version", "node", "element", "element_line_association", "face_element_side_association", "precision", "guid", "node_pair"}
    missing = required - set(counts)
    if missing:
        raise ValueError(f"PLY header is missing elements: {', '.join(sorted(missing))}")

    stream = BytesIO(data)
    stream.seek(body_offset)
    _read_exact(stream, counts["version"] * 4, "version")
    nodes = tuple(
        struct.unpack("<ddd", _read_exact(stream, 24, "node")) for _ in range(counts["node"])
    )
    for _ in range(counts["element"]):
        _read_exact(stream, 7, "element metadata")  # uchar, uchar, uchar, uint
        id_count = struct.unpack("<B", _read_exact(stream, 1, "element id count"))[0]
        _read_exact(stream, id_count * 4, "element ids")
    _read_exact(stream, counts["element_line_association"] * 8, "element-line associations")
    _read_exact(stream, counts["face_element_side_association"] * 8, "face associations")
    _read_exact(stream, counts["precision"] * 12, "precision")
    _read_exact(stream, counts["guid"], "guid")
    node_pairs: dict[int, tuple[int, int]] = {}
    for _ in range(counts["node_pair"]):
        owner, node1, node2 = struct.unpack("<III", _read_exact(stream, 12, "node pair"))
        if owner in node_pairs:
            raise ValueError(f"PLY has duplicate node_pair owner {owner}")
        node_pairs[owner] = (node1, node2)
    return PlyMesh(nodes=nodes, node_pairs=node_pairs)


def _read_exact(stream: BytesIO, size: int, label: str) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError(f"unexpected end of PLY while reading {label}")
    return value


def _point_close(left: tuple[float, float], right: tuple[float, float], tolerance: float) -> bool:
    return abs(left[0] - right[0]) <= tolerance and abs(left[1] - right[1]) <= tolerance
