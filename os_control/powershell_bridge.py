import os
import subprocess

from core.config import POWERSHELL_EXECUTABLE

# Vetted command templates only; no arbitrary user-provided PowerShell.
POWER_SHELL_TEMPLATES = {
    "open_app": {
        "script": "Start-Process -FilePath $env:JARVIS_APP_PATH",
        "env_keys": ("JARVIS_APP_PATH",),
    },
    "close_app": {
        "script": (
            "$p=$env:JARVIS_APP_PROCESS; "
            "$n=[System.IO.Path]::GetFileNameWithoutExtension($p); "
            "Stop-Process -Name $n -Force -ErrorAction Stop"
        ),
        "env_keys": ("JARVIS_APP_PROCESS",),
    },
    "shutdown": {
        "script": "shutdown /s /t 0",
        "env_keys": (),
    },
    "restart": {
        "script": "shutdown /r /t 0",
        "env_keys": (),
    },
    "sleep": {
        "script": (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "[System.Windows.Forms.Application]::SetSuspendState('Suspend', $false, $false)"
        ),
        "env_keys": (),
    },
    "lock": {
        "script": "rundll32.exe user32.dll,LockWorkStation",
        "env_keys": (),
    },
    "logoff": {
        "script": "shutdown /l",
        "env_keys": (),
    },
}


def run_template(template_name, env_overrides=None, timeout_seconds=30):
    template = POWER_SHELL_TEMPLATES.get(template_name)
    if not template:
        return False, f"Unknown PowerShell template: {template_name}", ""

    env = os.environ.copy()
    env_overrides = env_overrides or {}
    for required in template["env_keys"]:
        if required not in env_overrides:
            return False, f"Missing template parameter: {required}", ""
        env[required] = str(env_overrides[required])

    result = subprocess.run(
        [POWERSHELL_EXECUTABLE, "-NoProfile", "-NonInteractive", "-Command", template["script"]],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=env,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        return False, stderr or f"PowerShell template failed with code {result.returncode}", ""

    return True, "", (result.stdout or "").strip()
