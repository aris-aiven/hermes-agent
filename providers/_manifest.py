"""Manifest admission shared by model-provider directory import paths."""

from pathlib import Path

from hermes_cli.plugins_manifest import parse_manifest_file, requires_hermes_error


def directory_load_error(plugin_dir: Path, source: str) -> str | None:
    """Use native parsing and version policy; manifestless legacy directories remain supported."""
    for filename in ("plugin.yaml", "plugin.yml"):
        manifest_file = plugin_dir / filename
        if not manifest_file.is_file():
            continue
        manifest = parse_manifest_file(manifest_file, plugin_dir, source, "")
        if manifest is None:
            return f"invalid plugin manifest: {filename}"
        return requires_hermes_error(manifest)
    return None
