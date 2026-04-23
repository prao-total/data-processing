#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


INPUT_DIR = r"/path/to/gis_input_dir"
OUTPUT_DIR = r"/path/to/output_dir"
GIS_STEM = ""
OUTPUT_JSON = True

REQUIRED_EXTENSIONS = (".dat", ".id", ".ind", ".map", ".tab")
TAB_FILE_RE = re.compile(r'^File\s+"([^"]+)"', re.IGNORECASE)
TAB_CHARSET_RE = re.compile(r'^!charset\s+"([^"]+)"', re.IGNORECASE)
TAB_FIELDS_RE = re.compile(r"^Fields\s+(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class TabField:
    name: str
    definition: str


@dataclass
class TabMetadata:
    version: str | None = None
    charset: str | None = None
    table_type: str | None = None
    declared_files: list[str] = field(default_factory=list)
    fields_declared: int | None = None
    fields: list[TabField] = field(default_factory=list)


@dataclass(frozen=True)
class GisSystem:
    stem: str
    directory: Path
    files: dict[str, Path]


def discover_gis_system(input_dir: Path, requested_stem: str | None = None) -> GisSystem:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    grouped: dict[str, dict[str, Path]] = {}
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in REQUIRED_EXTENSIONS:
            continue
        grouped.setdefault(path.stem.lower(), {})[ext] = path

    if not grouped:
        raise FileNotFoundError(
            f"No GIS table files with extensions {REQUIRED_EXTENSIONS} were found in {input_dir}."
        )

    if requested_stem is not None:
        matches = grouped.get(requested_stem.lower())
        if matches is None:
            available = ", ".join(sorted(grouped))
            raise ValueError(
                f"No GIS system with stem '{requested_stem}' was found. Available stems: {available}"
            )
        missing = [ext for ext in REQUIRED_EXTENSIONS if ext not in matches]
        if missing:
            raise ValueError(
                f"GIS system '{requested_stem}' is incomplete. Missing files: {', '.join(missing)}"
            )
        return GisSystem(
            stem=matches[".tab"].stem,
            directory=input_dir.resolve(),
            files={ext: matches[ext] for ext in REQUIRED_EXTENSIONS},
        )

    complete_systems = [
        GisSystem(
            stem=files[".tab"].stem,
            directory=input_dir.resolve(),
            files={ext: files[ext] for ext in REQUIRED_EXTENSIONS},
        )
        for files in grouped.values()
        if all(ext in files for ext in REQUIRED_EXTENSIONS)
    ]

    if len(complete_systems) == 1:
        return complete_systems[0]

    if len(complete_systems) > 1:
        stems = ", ".join(sorted(system.stem for system in complete_systems))
        raise ValueError(
            f"Multiple complete GIS systems were found in {input_dir}: {stems}. "
            f"Set GIS_STEM at the top of the script to choose one."
        )

    incomplete = []
    for stem, files in sorted(grouped.items()):
        present = sorted(files)
        missing = [ext for ext in REQUIRED_EXTENSIONS if ext not in files]
        incomplete.append(
            f"{stem}: present={present}, missing={missing}"
        )
    raise ValueError(
        "No complete GIS system was found. Incomplete file groups detected: "
        + "; ".join(incomplete)
    )


def parse_tab_file(tab_path: Path) -> TabMetadata:
    metadata = TabMetadata()
    lines = tab_path.read_text(encoding="utf-8", errors="replace").splitlines()

    in_fields_block = False
    remaining_fields = 0

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if in_fields_block and remaining_fields > 0:
            if line.lower().startswith(("index ", "unique ", "type ", "file ", "begin ", "metadata ")):
                in_fields_block = False
                remaining_fields = 0
            else:
                parts = line.split(maxsplit=1)
                field_name = parts[0]
                field_definition = parts[1] if len(parts) > 1 else ""
                metadata.fields.append(TabField(name=field_name, definition=field_definition))
                remaining_fields -= 1
                if remaining_fields == 0:
                    in_fields_block = False
                continue

        lower_line = line.lower()
        if lower_line.startswith("!version"):
            metadata.version = line.split(maxsplit=1)[1] if len(line.split(maxsplit=1)) > 1 else None
            continue

        charset_match = TAB_CHARSET_RE.match(line)
        if charset_match:
            metadata.charset = charset_match.group(1)
            continue

        file_match = TAB_FILE_RE.match(line)
        if file_match:
            metadata.declared_files.append(file_match.group(1))
            continue

        if lower_line.startswith("type "):
            metadata.table_type = line.split(maxsplit=1)[1] if len(line.split(maxsplit=1)) > 1 else None
            continue

        fields_match = TAB_FIELDS_RE.match(line)
        if fields_match:
            metadata.fields_declared = int(fields_match.group(1))
            in_fields_block = True
            remaining_fields = metadata.fields_declared
            continue

    return metadata


def build_summary(system: GisSystem, metadata: TabMetadata) -> dict[str, object]:
    file_summary = {
        ext.lstrip(".").upper(): {
            "name": path.name,
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
        }
        for ext, path in system.files.items()
    }

    return {
        "stem": system.stem,
        "directory": str(system.directory),
        "files": file_summary,
        "tab_metadata": {
            "version": metadata.version,
            "charset": metadata.charset,
            "table_type": metadata.table_type,
            "declared_files": metadata.declared_files,
            "fields_declared": metadata.fields_declared,
            "fields_found": len(metadata.fields),
            "fields": [asdict(field) for field in metadata.fields],
        },
    }


def print_summary(summary: dict[str, object]) -> None:
    print(f"GIS system: {summary['stem']}")
    print(f"Directory: {summary['directory']}")
    print("")
    print("Files:")
    for extension, info in summary["files"].items():
        print(f"  {extension}: {info['name']} ({info['size_bytes']} bytes)")

    tab_metadata = summary["tab_metadata"]
    print("")
    print("TAB metadata:")
    print(f"  Version: {tab_metadata['version'] or 'unknown'}")
    print(f"  Charset: {tab_metadata['charset'] or 'unknown'}")
    print(f"  Type: {tab_metadata['table_type'] or 'unknown'}")

    declared_files = tab_metadata["declared_files"]
    if declared_files:
        print(f"  Declared files: {', '.join(declared_files)}")

    fields_declared = tab_metadata["fields_declared"]
    fields_found = tab_metadata["fields_found"]
    print(f"  Fields: {fields_found}" + (f" of {fields_declared} declared" if fields_declared is not None else ""))

    fields = tab_metadata["fields"]
    if fields:
        print("")
        print("Field definitions:")
        for field in fields:
            print(f"  {field['name']}: {field['definition']}")


def write_summary(summary: dict[str, object], output_dir: str, output_json: bool) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    suffix = "json" if output_json else "txt"
    output_path = destination / f"{summary['stem']}_gis_summary.{suffix}"

    if output_json:
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        lines = [
            f"GIS system: {summary['stem']}",
            f"Directory: {summary['directory']}",
            "",
            "Files:",
        ]
        for extension, info in summary["files"].items():
            lines.append(f"  {extension}: {info['name']} ({info['size_bytes']} bytes)")

        tab_metadata = summary["tab_metadata"]
        lines.extend(
            [
                "",
                "TAB metadata:",
                f"  Version: {tab_metadata['version'] or 'unknown'}",
                f"  Charset: {tab_metadata['charset'] or 'unknown'}",
                f"  Type: {tab_metadata['table_type'] or 'unknown'}",
            ]
        )

        declared_files = tab_metadata["declared_files"]
        if declared_files:
            lines.append(f"  Declared files: {', '.join(declared_files)}")

        fields_declared = tab_metadata["fields_declared"]
        fields_found = tab_metadata["fields_found"]
        field_line = f"  Fields: {fields_found}"
        if fields_declared is not None:
            field_line += f" of {fields_declared} declared"
        lines.append(field_line)

        fields = tab_metadata["fields"]
        if fields:
            lines.extend(["", "Field definitions:"])
            for field in fields:
                lines.append(f"  {field['name']}: {field['definition']}")

        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return output_path


def main() -> int:
    input_dir = Path(INPUT_DIR).expanduser()
    requested_stem = GIS_STEM.strip() or None

    try:
        system = discover_gis_system(input_dir, requested_stem)
        metadata = parse_tab_file(system.files[".tab"])
        summary = build_summary(system, metadata)
        output_path = write_summary(summary, OUTPUT_DIR, OUTPUT_JSON)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print_summary(summary)
    print("")
    print(f"Wrote summary to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
