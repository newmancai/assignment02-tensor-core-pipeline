"""Allowlisted Python/native execution identity receipts.

Recording only ``backend='evolution'`` or a top-level Python package path is
insufficient: dispatch can fall back, import order can select another extension,
and headers or shared objects can drift.  These receipts capture the effective
interpreter, ordered import path, imported files, native modules, include roots,
and explicitly supplied source/header dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig
from types import ModuleType
from typing import Iterable


RECORDED_ENVIRONMENT_KEYS = (
    "CUDA_HOME",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "PATH",
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileIdentity:
    path: str
    sha256: str | None
    exists: bool
    kind: str


@dataclass(frozen=True)
class ModuleIdentity:
    requested_name: str
    imported_name: str
    file: str | None
    sha256: str | None
    loader: str | None
    package_version: str | None
    native_extension: bool


@dataclass(frozen=True)
class ExecutionIdentityReceipt:
    python_executable: str
    python_version: str
    python_implementation: str
    python_prefix: str
    python_include_dirs: tuple[str, ...]
    extension_suffix: str | None
    soabi: str | None
    sys_path: tuple[str, ...]
    environment: dict[str, str | None]
    modules: tuple[ModuleIdentity, ...]
    loaded_native_extensions: tuple[ModuleIdentity, ...]
    dependency_files: tuple[FileIdentity, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def canonical_sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()


@lru_cache(maxsize=None)
def _root_distribution_version(root: str) -> str | None:
    try:
        packages_distributions = getattr(metadata, "packages_distributions", None)
        distributions = (
            packages_distributions().get(root, ())
            if packages_distributions is not None
            else ()
        )
        if distributions:
            return metadata.version(distributions[0])
        return metadata.version(root)
    except (metadata.PackageNotFoundError, ValueError):
        return None


def _distribution_version(module_name: str) -> str | None:
    return _root_distribution_version(module_name.split(".", 1)[0])


def module_identity(module: ModuleType, requested_name: str | None = None) -> ModuleIdentity:
    path_value = getattr(module, "__file__", None)
    path = Path(path_value).resolve() if path_value else None
    suffix = path.suffix if path is not None else ""
    native = suffix in {".so", ".pyd", ".dylib"} or ".so." in (path.name if path else "")
    loader = getattr(module, "__loader__", None)
    return ModuleIdentity(
        requested_name=requested_name or module.__name__,
        imported_name=module.__name__,
        file=str(path) if path else None,
        sha256=file_sha256(path) if path and path.is_file() else None,
        loader=type(loader).__name__ if loader is not None else None,
        package_version=_distribution_version(module.__name__),
        native_extension=native,
    )


def identify_dependency_file(path_value: str | Path) -> FileIdentity:
    path = Path(path_value).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".py":
        kind = "python_source"
    elif suffix in {".h", ".hh", ".hpp", ".cuh"}:
        kind = "native_header"
    elif suffix in {".c", ".cc", ".cpp", ".cu"}:
        kind = "native_translation_unit"
    elif suffix in {".so", ".pyd", ".dylib"}:
        kind = "native_extension"
    else:
        kind = "other"
    exists = path.is_file()
    return FileIdentity(
        path=str(path),
        sha256=file_sha256(path) if exists else None,
        exists=exists,
        kind=kind,
    )


def collect_execution_identity(
    module_names: Iterable[str] = (),
    dependency_files: Iterable[str | Path] = (),
) -> ExecutionIdentityReceipt:
    requested = []
    for name in module_names:
        requested.append(module_identity(importlib.import_module(name), name))
    loaded = []
    seen_files = set()
    for name, module in sorted(sys.modules.items()):
        if module is None or not getattr(module, "__file__", None):
            continue
        identity = module_identity(module, name)
        if identity.native_extension and identity.file not in seen_files:
            loaded.append(identity)
            seen_files.add(identity.file)
    include_dirs = tuple(
        dict.fromkeys(
            str(Path(value).resolve())
            for key in ("include", "platinclude")
            if (value := sysconfig.get_paths().get(key))
        )
    )
    return ExecutionIdentityReceipt(
        python_executable=str(Path(sys.executable).resolve()),
        python_version=sys.version,
        python_implementation=platform.python_implementation(),
        python_prefix=str(Path(sys.prefix).resolve()),
        python_include_dirs=include_dirs,
        extension_suffix=sysconfig.get_config_var("EXT_SUFFIX"),
        soabi=sysconfig.get_config_var("SOABI"),
        sys_path=tuple(sys.path),
        environment={key: os.environ.get(key) for key in RECORDED_ENVIRONMENT_KEYS},
        modules=tuple(requested),
        loaded_native_extensions=tuple(loaded),
        dependency_files=tuple(identify_dependency_file(path) for path in dependency_files),
    )


def write_execution_identity(receipt: ExecutionIdentityReceipt, path: Path) -> None:
    payload = receipt.to_dict()
    payload["canonical_sha256"] = receipt.canonical_sha256()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
