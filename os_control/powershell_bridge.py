import os
import subprocess

from core.config import POWERSHELL_EXECUTABLE

# Vetted command templates only; no arbitrary user-provided PowerShell.
POWER_SHELL_TEMPLATES = {
    "open_app": {
        "script": (
            "$p=$env:JARVIS_APP_PATH; "
            "if ($p -like 'shell:*') { Start-Process $p } "
            "else { Start-Process -FilePath $p }"
        ),
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
    "volume_up": {
        "script": (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "[System.Windows.Forms.SendKeys]::SendWait([char]175)"
        ),
        "env_keys": (),
    },
    "volume_down": {
        "script": (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "[System.Windows.Forms.SendKeys]::SendWait([char]174)"
        ),
        "env_keys": (),
    },
    "volume_mute": {
        "script": (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "[System.Windows.Forms.SendKeys]::SendWait([char]173)"
        ),
        "env_keys": (),
    },
    "volume_set": {
        "script": (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$target=[int]$env:JARVIS_VOLUME_PERCENT; "
            "$target=[Math]::Max(0,[Math]::Min(100,$target)); "
            "for($i=0;$i -lt 55;$i++){[System.Windows.Forms.SendKeys]::SendWait([char]174)}; "
            "$steps=[Math]::Round($target/2); "
            "for($i=0;$i -lt $steps;$i++){[System.Windows.Forms.SendKeys]::SendWait([char]175)}; "
            "Write-Output ('volume_set=' + $target)"
        ),
        "env_keys": ("JARVIS_VOLUME_PERCENT",),
    },
    "brightness_up": {
        "script": (
            "$b=Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness; "
            "$m=Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods; "
            "$target=[Math]::Min(100,[int]$b.CurrentBrightness + 10); "
            "$m.WmiSetBrightness(1,$target)"
        ),
        "env_keys": (),
    },
    "brightness_down": {
        "script": (
            "$b=Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness; "
            "$m=Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods; "
            "$target=[Math]::Max(10,[int]$b.CurrentBrightness - 10); "
            "$m.WmiSetBrightness(1,$target)"
        ),
        "env_keys": (),
    },
    "brightness_set": {
        "script": (
            "$target=[int]$env:JARVIS_BRIGHTNESS_PERCENT; "
            "$target=[Math]::Max(0,[Math]::Min(100,$target)); "
            "$m=Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods; "
            "if(-not $m){ throw 'Brightness control unavailable' }; "
            "$m.WmiSetBrightness(1,$target); "
            "Write-Output ('brightness_set=' + $target)"
        ),
        "env_keys": ("JARVIS_BRIGHTNESS_PERCENT",),
    },
    "wifi_on": {
        "script": (
            "$i=Get-NetAdapter | Where-Object { $_.Name -match 'Wi-?Fi|Wireless' } | Select-Object -First 1; "
            "if(-not $i){ throw 'No Wi-Fi adapter found' }; "
            "Enable-NetAdapter -Name $i.Name -Confirm:$false"
        ),
        "env_keys": (),
    },
    "wifi_off": {
        "script": (
            "$i=Get-NetAdapter | Where-Object { $_.Name -match 'Wi-?Fi|Wireless' } | Select-Object -First 1; "
            "if(-not $i){ throw 'No Wi-Fi adapter found' }; "
            "Disable-NetAdapter -Name $i.Name -Confirm:$false"
        ),
        "env_keys": (),
    },
    "bluetooth_on": {
        "script": (
            "$d=Get-PnpDevice -Class Bluetooth -ErrorAction SilentlyContinue; "
            "if(-not $d){ throw 'No Bluetooth device found' }; "
            "$d | ForEach-Object { Enable-PnpDevice -InstanceId $_.InstanceId -Confirm:$false }"
        ),
        "env_keys": (),
    },
    "bluetooth_off": {
        "script": (
            "$d=Get-PnpDevice -Class Bluetooth -ErrorAction SilentlyContinue; "
            "if(-not $d){ throw 'No Bluetooth device found' }; "
            "$d | ForEach-Object { Disable-PnpDevice -InstanceId $_.InstanceId -Confirm:$false }"
        ),
        "env_keys": (),
    },
    "notifications_on": {
        "script": (
            "$path='HKCU:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Notifications\\Settings'; "
            "if(-not (Test-Path $path)){ New-Item -Path $path -Force | Out-Null }; "
            "Set-ItemProperty -Path $path -Name 'NOC_GLOBAL_SETTING_TOASTS_ENABLED' -Type DWord -Value 1; "
            "Write-Output 'notifications_on'"
        ),
        "env_keys": (),
    },
    "notifications_off": {
        "script": (
            "$path='HKCU:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Notifications\\Settings'; "
            "if(-not (Test-Path $path)){ New-Item -Path $path -Force | Out-Null }; "
            "Set-ItemProperty -Path $path -Name 'NOC_GLOBAL_SETTING_TOASTS_ENABLED' -Type DWord -Value 0; "
            "Write-Output 'notifications_off'"
        ),
        "env_keys": (),
    },
    "screenshot": {
        "script": (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "Add-Type -AssemblyName System.Drawing; "
            "$bounds=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
            "$bmp=New-Object System.Drawing.Bitmap $bounds.Width,$bounds.Height; "
            "$g=[System.Drawing.Graphics]::FromImage($bmp); "
            "$g.CopyFromScreen($bounds.Location,[System.Drawing.Point]::Empty,$bounds.Size); "
            "$dir=[Environment]::GetFolderPath('Desktop'); "
            "$path=Join-Path $dir ('jarvis_shot_{0}.png' -f (Get-Date -Format 'yyyyMMdd_HHmmss')); "
            "$bmp.Save($path,[System.Drawing.Imaging.ImageFormat]::Png); "
            "$g.Dispose(); $bmp.Dispose(); "
            "Write-Output $path"
        ),
        "env_keys": (),
    },
    "empty_recycle_bin": {
        "script": "Clear-RecycleBin -Force -ErrorAction Stop",
        "env_keys": (),
    },
    "list_processes": {
        "script": (
            "Get-Process | Sort-Object CPU -Descending | "
            "Select-Object -First 12 -ExpandProperty ProcessName | Out-String"
        ),
        "env_keys": (),
    },
    "focus_window": {
        "script": (
            "$q=$env:JARVIS_WINDOW_QUERY; "
            "if(-not $q){ throw 'Missing window query' }; "
            "$shell=New-Object -ComObject WScript.Shell; "
            "$ok=$shell.AppActivate($q); "
            "if(-not $ok){ throw ('No matching window for: ' + $q) }; "
            "Write-Output ('focused=' + $q)"
        ),
        "env_keys": ("JARVIS_WINDOW_QUERY",),
    },
    "window_maximize": {
        "script": (
            "$shell=New-Object -ComObject WScript.Shell; "
            "$shell.SendKeys('%{SPACE}'); "
            "$shell.SendKeys('x'); "
            "Write-Output 'window_maximized'"
        ),
        "env_keys": (),
    },
    "window_minimize": {
        "script": (
            "$shell=New-Object -ComObject WScript.Shell; "
            "$shell.SendKeys('%{SPACE}'); "
            "$shell.SendKeys('n'); "
            "Write-Output 'window_minimized'"
        ),
        "env_keys": (),
    },
    "window_snap_left": {
        "script": (
            "$shell=New-Object -ComObject WScript.Shell; "
            "$shell.SendKeys('#{LEFT}'); "
            "Write-Output 'window_snap_left'"
        ),
        "env_keys": (),
    },
    "window_snap_right": {
        "script": (
            "$shell=New-Object -ComObject WScript.Shell; "
            "$shell.SendKeys('#{RIGHT}'); "
            "Write-Output 'window_snap_right'"
        ),
        "env_keys": (),
    },
    "window_next": {
        "script": (
            "$shell=New-Object -ComObject WScript.Shell; "
            "$shell.SendKeys('%{TAB}'); "
            "Write-Output 'window_next'"
        ),
        "env_keys": (),
    },
    "window_close_active": {
        "script": (
            "$shell=New-Object -ComObject WScript.Shell; "
            "$shell.SendKeys('%{F4}'); "
            "Write-Output 'window_closed'"
        ),
        "env_keys": (),
    },
    "media_play_pause": {
        "script": (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "[System.Windows.Forms.SendKeys]::SendWait([char]179); "
            "Write-Output 'media_play_pause'"
        ),
        "env_keys": (),
    },
    "media_next_track": {
        "script": (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "[System.Windows.Forms.SendKeys]::SendWait([char]176); "
            "Write-Output 'media_next_track'"
        ),
        "env_keys": (),
    },
    "media_previous_track": {
        "script": (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "[System.Windows.Forms.SendKeys]::SendWait([char]177); "
            "Write-Output 'media_previous_track'"
        ),
        "env_keys": (),
    },
    "media_stop": {
        "script": (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "[System.Windows.Forms.SendKeys]::SendWait([char]178); "
            "Write-Output 'media_stop'"
        ),
        "env_keys": (),
    },
    "media_seek_forward": {
        "script": (
            "$s=[int]$env:JARVIS_MEDIA_SEEK_SECONDS; "
            "$steps=[Math]::Max(1,[Math]::Ceiling($s/5.0)); "
            "$shell=New-Object -ComObject WScript.Shell; "
            "for($i=0;$i -lt $steps;$i++){ $shell.SendKeys('^{RIGHT}') }; "
            "Write-Output ('media_seek_forward=' + $s)"
        ),
        "env_keys": ("JARVIS_MEDIA_SEEK_SECONDS",),
    },
    "media_seek_backward": {
        "script": (
            "$s=[int]$env:JARVIS_MEDIA_SEEK_SECONDS; "
            "$steps=[Math]::Max(1,[Math]::Ceiling($s/5.0)); "
            "$shell=New-Object -ComObject WScript.Shell; "
            "for($i=0;$i -lt $steps;$i++){ $shell.SendKeys('^{LEFT}') }; "
            "Write-Output ('media_seek_backward=' + $s)"
        ),
        "env_keys": ("JARVIS_MEDIA_SEEK_SECONDS",),
    },
    "browser_new_tab": {
        "script": (
            "$shell=New-Object -ComObject WScript.Shell; "
            "$shell.SendKeys('^t'); "
            "Write-Output 'browser_new_tab'"
        ),
        "env_keys": (),
    },
    "browser_close_tab": {
        "script": (
            "$shell=New-Object -ComObject WScript.Shell; "
            "$shell.SendKeys('^w'); "
            "Write-Output 'browser_close_tab'"
        ),
        "env_keys": (),
    },
    "browser_back": {
        "script": (
            "$shell=New-Object -ComObject WScript.Shell; "
            "$shell.SendKeys('%{LEFT}'); "
            "Write-Output 'browser_back'"
        ),
        "env_keys": (),
    },
    "browser_forward": {
        "script": (
            "$shell=New-Object -ComObject WScript.Shell; "
            "$shell.SendKeys('%{RIGHT}'); "
            "Write-Output 'browser_forward'"
        ),
        "env_keys": (),
    },
    "browser_open_url": {
        "script": (
            "$u=$env:JARVIS_BROWSER_URL; "
            "if(-not $u){ throw 'Missing browser URL' }; "
            "Start-Process -FilePath $u; "
            "Write-Output ('browser_open_url=' + $u)"
        ),
        "env_keys": ("JARVIS_BROWSER_URL",),
    },
    "browser_search_web": {
        "script": (
            "$q=$env:JARVIS_BROWSER_QUERY; "
            "if(-not $q){ throw 'Missing browser query' }; "
            "$url='https://www.google.com/search?q=' + [uri]::EscapeDataString($q); "
            "Start-Process -FilePath $url; "
            "Write-Output ('browser_search_web=' + $q)"
        ),
        "env_keys": ("JARVIS_BROWSER_QUERY",),
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
