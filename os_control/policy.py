import os
import threading

from core.config import (
    POLICY_ALLOW_READ_OUTSIDE_ALLOWLIST,
    POLICY_ALLOWED_PATHS,
    POLICY_BLOCKED_PATH_PREFIXES,
    POLICY_COMMAND_PERMISSIONS,
    POLICY_PROFILES,
    POLICY_READ_ONLY_MODE,
)


class PolicyEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._allowed_roots = tuple(os.path.abspath(p) for p in POLICY_ALLOWED_PATHS)
        self._blocked_prefixes = tuple(os.path.abspath(p) for p in POLICY_BLOCKED_PATH_PREFIXES)
        self._profiles = POLICY_PROFILES
        self._active_profile = "normal"
        self._read_only_mode = POLICY_READ_ONLY_MODE
        self._command_permissions = dict(POLICY_COMMAND_PERMISSIONS)

    def set_read_only_mode(self, enabled):
        with self._lock:
            self._read_only_mode = bool(enabled)

    def set_command_permission(self, command_key, enabled):
        with self._lock:
            self._command_permissions[command_key] = bool(enabled)

    def is_command_allowed(self, command_key):
        with self._lock:
            return bool(self._command_permissions.get(command_key, False))

    def can_access_path(self, path, write=False):
        target = os.path.abspath(path)

        for blocked in self._blocked_prefixes:
            if target == blocked or target.startswith(blocked + os.sep):
                return False, f"Blocked by policy: {target}"

        if not any(
            target == root or target.startswith(root + os.sep)
            for root in self._allowed_roots
        ):
            if write or not bool(POLICY_ALLOW_READ_OUTSIDE_ALLOWLIST):
                return False, f"Path outside allowlist: {target}"

        with self._lock:
            if write and self._read_only_mode:
                return False, "Policy read-only mode is enabled."

        return True, ""

    def set_profile(self, profile_name):
        profile_key = profile_name.lower().strip()
        profile = self._profiles.get(profile_key)
        if not profile:
            return False, f"Unknown profile: {profile_name}"

        with self._lock:
            self._active_profile = profile_key
            self._read_only_mode = bool(profile.get("read_only_mode", False))
            self._command_permissions = dict(profile.get("command_permissions", {}))
        return True, f"Policy profile set to: {profile_key}"

    def get_profile(self):
        with self._lock:
            return self._active_profile

    def status(self):
        with self._lock:
            permissions = dict(self._command_permissions)
            profile = self._active_profile
            read_only = self._read_only_mode
        return {
            "profile": profile,
            "read_only_mode": read_only,
            "allowed_paths": list(self._allowed_roots),
            "blocked_prefixes": list(self._blocked_prefixes),
            "permissions": permissions,
        }


policy_engine = PolicyEngine()
