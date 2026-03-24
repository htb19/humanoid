from __future__ import annotations

from pathlib import Path

from .config import describe_file


def _load_urdf_module():
    try:
        from isaacsim.asset.importer.urdf import _urdf
    except ImportError:
        from omni.importer.urdf import _urdf
    return _urdf


def build_import_config():
    _urdf = _load_urdf_module()
    import_config = _urdf.ImportConfig()
    # Keep the config minimal while debugging import failures.
    import_config.set_fix_base(True)
    return import_config


def verify_runtime_urdf(urdf_path: Path) -> None:
    description = describe_file(urdf_path)
    print("[urdf] runtime file:", description)
    if not description["exists"]:
        raise FileNotFoundError(f"Runtime URDF does not exist: {urdf_path}")
    if not description["is_file"]:
        raise FileNotFoundError(f"Runtime URDF path is not a file: {urdf_path}")
    if not description["readable"]:
        raise PermissionError(f"Runtime URDF is not readable: {urdf_path}")


def import_urdf(urdf_path: Path):
    verify_runtime_urdf(urdf_path)
    import_config = build_import_config()
    _urdf = _load_urdf_module()
    urdf_interface = _urdf.acquire_urdf_interface()
    asset_root = str(urdf_path.parent)
    asset_name = urdf_path.name

    print(
        "[urdf] importing with args:",
        {
            "asset_root": asset_root,
            "asset_name": asset_name,
            "import_config_type": type(import_config).__name__,
        },
    )
    parsed_robot = urdf_interface.parse_urdf(asset_root, asset_name, import_config)
    print("[urdf] parsed robot type:", type(parsed_robot).__name__)

    imported_prim_path = urdf_interface.import_robot(
        asset_root,
        asset_name,
        parsed_robot,
        import_config,
        "",
    )
    print("[urdf] import result:", {"prim_path": imported_prim_path})
    if not imported_prim_path:
        raise RuntimeError(f"Failed to import URDF: {urdf_path}")
    return imported_prim_path
