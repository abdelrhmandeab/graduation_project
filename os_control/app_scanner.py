"""Installed app discovery for Windows.

The scanner builds the same catalog shape used by os_control.app_ops:

    {
        "app.exe": {
            "canonical_name": "App Name",
            "aliases": ["app name", "short alias"],
        },
    }

Discovery is cached under .jarvis_cache/app_catalog.json and refreshed when the
cache is older than 24 hours or when force=True.
"""

from __future__ import annotations

import json
import os
import platform
import re
import time
from pathlib import Path
from typing import Iterable

from core.logger import logger

_IS_WINDOWS = platform.system().lower() == "windows"
_CACHE_TTL_SECONDS = 24 * 60 * 60
_CACHE_PATH = Path(__file__).resolve().parents[1] / ".jarvis_cache" / "app_catalog.json"

try:
    import winreg
except ImportError:  # pragma: no cover - non-Windows fallback
    winreg = None

try:
    import win32com.client
except ImportError:  # pragma: no cover - optional dependency
    win32com = None

_VERSION_RE = re.compile(r"\b(?:v)?\d+(?:\.\d+)*\b", re.IGNORECASE)
_NON_WORD_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_COMPANY_PREFIXES = (
    "adobe",
    "apple",
    "audacity",
    "autodesk",
    "google",
    "ibm",
    "intel",
    "microsoft",
    "mozilla",
    "oracle",
    "reaper",
    "samsung",
    "slack",
    "the",
)


def _normalize_alias(text: str) -> str:
    value = _WS_RE.sub(" ", _NON_WORD_RE.sub(" ", (text or "").lower())).strip()
    return value


def _dedupe_aliases(aliases: Iterable[str]) -> list[str]:
    ordered = []
    seen = set()
    for alias in aliases:
        value = _normalize_alias(alias)
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _strip_version_tokens(text: str) -> str:
    return _WS_RE.sub(" ", _VERSION_RE.sub(" ", text or "")).strip()


def _strip_company_prefixes(text: str) -> str:
    value = (text or "").strip()
    lowered = value.lower()
    for prefix in _COMPANY_PREFIXES:
        if lowered.startswith(prefix + " "):
            return value[len(prefix) :].strip()
    return value


def _display_name_variants(display_name: str) -> list[str]:
    value = _normalize_alias(display_name)
    if not value:
        return []

    variants = [value]
    no_version = _normalize_alias(_strip_version_tokens(value))
    if no_version and no_version not in variants:
        variants.append(no_version)

    company_stripped = _normalize_alias(_strip_company_prefixes(value))
    if company_stripped and company_stripped not in variants:
        variants.append(company_stripped)

    company_and_version = _normalize_alias(_strip_version_tokens(_strip_company_prefixes(value)))
    if company_and_version and company_and_version not in variants:
        variants.append(company_and_version)

    tokens = value.split()
    if len(tokens) > 1:
        last_token = tokens[-1]
        if last_token not in variants:
            variants.append(last_token)

    return _dedupe_aliases(variants)


def _guess_executable_key(display_name: str, install_location: str = "", display_icon: str = "", target_path: str = "") -> str:
    candidates = [target_path, display_icon, install_location, display_name]
    for candidate in candidates:
        value = str(candidate or "").strip()
        if not value:
            continue
        value = value.split(",")[0].strip().strip('"')
        if value.lower().startswith(("shell:", "ms-", "http://", "https://")):
            continue
        if ".exe" in value.lower():
            base = Path(value).name
            if base:
                return base.lower()
    return ""


def _extract_path_from_text(text: str) -> str:
    value = str(text or "").strip().strip('"')
    if not value:
        return ""
    value = value.split(",")[0].strip()
    if value.lower().endswith(".exe") or ".exe " in value.lower():
        return value.split(" ")[0].strip('"')
    return value


def _read_reg_value(key, name: str):
    try:
        if name is None:
            try:
                return winreg.QueryValue(key, "")
            except Exception:
                return winreg.QueryValue(key, None)
        return winreg.QueryValueEx(key, name)[0]
    except Exception:
        return None


def _scan_uninstall_registry() -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    if not _IS_WINDOWS or winreg is None:
        return catalog

    hives = (
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall", getattr(winreg, "KEY_WOW64_64KEY", 0)),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall", getattr(winreg, "KEY_WOW64_32KEY", 0)),
        # User-installed apps (Discord, Spotify, etc.) register here
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall", 0),
    )
    for hive, path, extra_flags in hives:
        try:
            with winreg.OpenKey(hive, path, 0, winreg.KEY_READ | extra_flags) as root:
                sub_count = winreg.QueryInfoKey(root)[0]
                for index in range(sub_count):
                    try:
                        sub_name = winreg.EnumKey(root, index)
                        with winreg.OpenKey(root, sub_name) as subkey:
                            display_name = _read_reg_value(subkey, "DisplayName")
                            if not display_name:
                                continue
                            install_location = str(_read_reg_value(subkey, "InstallLocation") or "").strip()
                            display_icon = str(_read_reg_value(subkey, "DisplayIcon") or "").strip()
                            uninstall_string = str(_read_reg_value(subkey, "UninstallString") or "").strip()
                            target_path = _extract_path_from_text(display_icon) or _extract_path_from_text(uninstall_string)
                            executable = _guess_executable_key(display_name, install_location, display_icon, target_path)
                            if not executable:
                                continue
                            aliases = _display_name_variants(str(display_name))
                            if install_location:
                                aliases.append(_normalize_alias(Path(install_location).name))
                            full_path = target_path if target_path and target_path.lower().endswith(".exe") else ""
                            catalog[executable] = {
                                "canonical_name": str(display_name).strip(),
                                "aliases": _dedupe_aliases(aliases),
                                "path": full_path,
                            }
                    except Exception:
                        continue
        except Exception:
            continue
    return catalog


def _resolve_start_menu_shortcut(link_path: Path) -> tuple[str, str]:
    if win32com is None:
        return "", ""
    try:
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(str(link_path))
        target = str(getattr(shortcut, "TargetPath", "") or "").strip()
        arguments = str(getattr(shortcut, "Arguments", "") or "").strip()
        return target, arguments
    except Exception as exc:
        logger.debug("Shortcut parse failed for %s: %s", link_path, exc)
        return "", ""


def _scan_start_menu() -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    if not _IS_WINDOWS:
        return catalog

    start_menu = Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    if not start_menu.exists():
        return catalog

    for link_path in start_menu.rglob("*.lnk"):
        try:
            target_path, _arguments = _resolve_start_menu_shortcut(link_path)
            display_name = link_path.stem
            executable = _guess_executable_key(display_name, target_path=target_path)
            if not executable:
                continue
            aliases = _display_name_variants(display_name)
            target_name = Path(target_path).stem if target_path else ""
            if target_name:
                aliases.append(_normalize_alias(target_name))
            full_path = target_path if target_path and target_path.lower().endswith(".exe") else ""
            catalog[executable] = {
                "canonical_name": display_name,
                "aliases": _dedupe_aliases(aliases),
                "path": full_path,
            }
        except Exception:
            continue
    return catalog


def _scan_app_paths_registry() -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    if not _IS_WINDOWS or winreg is None:
        return catalog

    hives = (
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths", getattr(winreg, "KEY_WOW64_64KEY", 0)),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths", getattr(winreg, "KEY_WOW64_32KEY", 0)),
        # Per-user app registrations (Discord, Spotify, etc.)
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths", 0),
    )
    for hive, path, extra_flags in hives:
        try:
            with winreg.OpenKey(hive, path, 0, winreg.KEY_READ | extra_flags) as root:
                sub_count = winreg.QueryInfoKey(root)[0]
                for index in range(sub_count):
                    try:
                        sub_name = winreg.EnumKey(root, index)
                        with winreg.OpenKey(root, sub_name) as subkey:
                            default_path = str(_read_reg_value(subkey, None) or "").strip()
                            executable = _guess_executable_key(sub_name, target_path=default_path)
                            if not executable:
                                continue
                            canonical_name = Path(sub_name).stem if sub_name.lower().endswith(".exe") else sub_name
                            aliases = _display_name_variants(canonical_name)
                            if default_path:
                                aliases.append(_normalize_alias(Path(default_path).stem))
                            full_path = default_path if default_path and default_path.lower().endswith(".exe") else ""
                            catalog[executable] = {
                                "canonical_name": canonical_name,
                                "aliases": _dedupe_aliases(aliases),
                                "path": full_path,
                            }
                    except Exception:
                        continue
        except Exception:
            continue
    return catalog


def _load_cache() -> dict | None:
    try:
        if not _CACHE_PATH.exists():
            return None
        with _CACHE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return None
        timestamp = float(data.get("timestamp") or 0.0)
        catalog = data.get("catalog")
        if not isinstance(catalog, dict):
            return None
        if time.time() - timestamp > _CACHE_TTL_SECONDS:
            return None
        return {"timestamp": timestamp, "catalog": catalog}
    except Exception as exc:
        logger.debug("App catalog cache load failed: %s", exc)
        return None


def _save_cache(catalog: dict[str, dict]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {"timestamp": time.time(), "catalog": catalog}
        with _CACHE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.debug("App catalog cache write failed: %s", exc)


def _merge_catalogs(base_catalog: dict[str, dict] | None, discovered_catalog: dict[str, dict]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    base_catalog = base_catalog or {}

    for executable, payload in discovered_catalog.items():
        merged[executable.lower()] = {
            "canonical_name": str(payload.get("canonical_name") or executable).strip(),
            "aliases": _dedupe_aliases(payload.get("aliases") or []),
            "path": str(payload.get("path") or "").strip(),
        }

    for executable, payload in base_catalog.items():
        key = executable.lower()
        base_aliases = _dedupe_aliases(payload.get("aliases") or [])
        if key in merged:
            merged[key]["canonical_name"] = str(payload.get("canonical_name") or merged[key]["canonical_name"]).strip()
            merged[key]["aliases"] = _dedupe_aliases(list(base_aliases) + list(merged[key]["aliases"]))
            # Prefer discovered path (more accurate); base catalog rarely has a path field
            if not merged[key]["path"]:
                merged[key]["path"] = str(payload.get("path") or "").strip()
        else:
            merged[key] = {
                "canonical_name": str(payload.get("canonical_name") or executable).strip(),
                "aliases": base_aliases,
                "path": str(payload.get("path") or "").strip(),
            }

    return dict(sorted(merged.items(), key=lambda item: item[0]))


def _scan_store_apps() -> dict[str, dict]:
    """Scan Microsoft Store (UWP) apps via Get-StartApps PowerShell cmdlet."""
    catalog: dict[str, dict] = {}
    if not _IS_WINDOWS:
        return catalog
    try:
        import subprocess
        result = subprocess.run(
            [
                "powershell",
                "-NonInteractive",
                "-NoProfile",
                "-Command",
                "Get-StartApps | Select-Object Name, AppID | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=0x08000000,  # CREATE_NO_WINDOW
        )
        raw = (result.stdout or "").strip()
        if not raw:
            return catalog
        apps = json.loads(raw)
        if isinstance(apps, dict):
            apps = [apps]
        for app in apps:
            name = str(app.get("Name") or "").strip()
            app_id = str(app.get("AppID") or "").strip()
            if not name or not app_id:
                continue
            app_id_lower = app_id.lower()
            if app_id_lower.endswith((".exe", ".lnk")):
                # Classic Win32 — already covered by registry scanners
                continue
            # UWP: AppID looks like "Publisher.AppName_hash!App"
            if "_" not in app_id and "!" not in app_id:
                continue
            pkg_part = app_id.split("_")[0]
            short_name = pkg_part.split(".")[-1].lower()
            if not short_name:
                continue
            executable = f"{short_name}.uwp"
            aliases = _display_name_variants(name)
            aliases.append(_normalize_alias(short_name))
            catalog[executable] = {
                "canonical_name": name,
                "aliases": _dedupe_aliases(aliases),
                "path": f"shell:AppsFolder\\{app_id}",
            }
    except Exception as exc:
        logger.debug("Store app scan failed: %s", exc)
    return catalog


def _scan_installed_apps() -> dict[str, dict]:
    discovered = {}
    discovered.update(_scan_uninstall_registry())
    discovered.update(_scan_start_menu())
    discovered.update(_scan_app_paths_registry())
    discovered.update(_scan_store_apps())
    return discovered


def scan_installed_apps(base_catalog: dict | None = None, force: bool = False) -> dict:
    """Scan installed apps and return a merged catalog.

    Args:
        base_catalog: Hardcoded app catalog that should win for canonical names
            and aliases when an executable already exists in the dynamic scan.
        force: Ignore the cache and perform a fresh scan.
    """
    cached = None if force else _load_cache()
    if cached is not None:
        return _merge_catalogs(base_catalog, cached["catalog"])

    discovered = _scan_installed_apps()
    if discovered:
        _save_cache(discovered)
    elif cached is not None:
        discovered = cached["catalog"]
    return _merge_catalogs(base_catalog, discovered)
