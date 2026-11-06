from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .model import DesignVariables


@dataclass(slots=True)
class TemplateArchive:
    source: Path
    xml_name: str
    xml_bytes: bytes
    assets: dict[str, bytes]
    reference_design: DesignVariables | None = None

    @classmethod
    def load(cls, path: Path) -> TemplateArchive:
        path = path.resolve()
        if path.is_dir():
            xml_files = sorted(path.glob("*.xml"))
            if len(xml_files) != 1:
                raise ValueError(f"template directory must contain exactly one XML file: {path}")
            xml_path = xml_files[0]
            parameters_path = path / "parameters.json"
            assets = {
                item.relative_to(path).as_posix(): item.read_bytes()
                for item in path.rglob("*")
                if item.is_file()
                and item not in {xml_path, parameters_path}
                and not _is_generated_asset(item.relative_to(path).as_posix())
            }
            return cls(
                path,
                xml_path.name,
                xml_path.read_bytes(),
                assets,
                _read_reference_design(parameters_path),
            )

        if path.suffix.lower() != ".gsz":
            raise ValueError("template must be a directory or a .gsz file")
        with zipfile.ZipFile(path) as archive:
            root_files = [name for name in archive.namelist() if "/" not in name.rstrip("/")]
            xml_files = [name for name in root_files if name.lower().endswith(".xml")]
            if len(xml_files) != 1:
                raise ValueError(f"template GSZ must contain exactly one root XML file: {path}")
            xml_name = xml_files[0]
            # A solved GSZ can contain hundreds of megabytes of result tables in
            # nested folders.  They belong to the baseline run, not to the new
            # template, and GeoCmd will create fresh results after remeshing.
            assets = {
                name: archive.read(name)
                for name in root_files
                if name != xml_name
                and not name.endswith("/")
                and not _is_generated_asset(name)
            }
            return cls(
                path,
                xml_name,
                archive.read(xml_name),
                assets,
                _read_reference_design(path.parent / "parameters.json"),
            )

    def root(self) -> ET.Element:
        return ET.fromstring(self.xml_bytes)

    def write_gsz(self, path: Path, xml_bytes: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(f"{path.stem}.xml", xml_bytes)
            for name, content in sorted(self.assets.items()):
                archive.writestr(name, content)


def xml_bytes(root: ET.Element) -> bytes:
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _read_reference_design(path: Path) -> DesignVariables | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        parameters = payload.get("parameters", payload)
        mapping = {
            field: str(parameters[field])
            for field in DesignVariables.CSV_FIELDS
            if field != "secondary_right_slope_distance"
        }
        if "secondary_right_slope_distance" in parameters:
            mapping["secondary_right_slope_distance"] = str(
                parameters["secondary_right_slope_distance"]
            )
        else:
            old_clearance = float(parameters["secondary_bottom_clearance"])
            downstream_slope = float(parameters["downstream_slope"])
            mapping["secondary_right_slope_distance"] = str(
                old_clearance / math.sqrt(1.0 + downstream_slope**2)
            )
        design = DesignVariables.from_mapping(mapping)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid template reference parameters: {path}: {error}") from error
    errors = design.validate_bounds()
    if errors:
        raise ValueError(
            f"invalid template reference parameters: {path}: {'; '.join(errors)}"
        )
    return design


def _is_generated_asset(name: str) -> bool:
    """Return whether an archive member belongs to an old mesh or solve."""
    normalized = name.replace("\\", "/").strip("/")
    basename = normalized.rsplit("/", 1)[-1]
    if re.fullmatch(r"mesh_\d+\.ply", basename, flags=re.IGNORECASE):
        return True
    generated_directories = {
        "analyses",
        "results",
        "solver",
        "tables",
    }
    return any(part.lower() in generated_directories for part in normalized.split("/")[:-1])
