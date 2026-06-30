use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    Manager, PhysicalPosition, WindowEvent,
};

#[tauri::command]
fn open_dashboard(app: tauri::AppHandle) {
    if let Some(w) = app.get_webview_window("dashboard") {
        let _ = w.show();
        let _ = w.set_focus();
    }
}

#[tauri::command]
fn hide_dashboard(app: tauri::AppHandle) {
    if let Some(w) = app.get_webview_window("dashboard") {
        let _ = w.hide();
    }
}

#[tauri::command]
fn quit_app(app: tauri::AppHandle) {
    app.exit(0);
}

#[tauri::command]
fn show_overlay(app: tauri::AppHandle) {
    if let Some(w) = app.get_webview_window("overlay") {
        position_overlay(&w);
        let _ = w.show();
        let _ = w.set_focus();
    }
}

#[tauri::command]
fn hide_overlay(app: tauri::AppHandle) {
    if let Some(w) = app.get_webview_window("overlay") {
        let _ = w.hide();
    }
}

/// Position the frameless overlay near the bottom-right of the primary monitor.
fn position_overlay(window: &tauri::WebviewWindow) {
    if let Ok(Some(monitor)) = window.primary_monitor() {
        let screen = monitor.size();
        let win = window.outer_size().unwrap_or_default();
        let margin: i32 = 24;
        let taskbar_gap: i32 = 56;
        let x = screen.width as i32 - win.width as i32 - margin;
        let y = screen.height as i32 - win.height as i32 - margin - taskbar_gap;
        let _ = window.set_position(PhysicalPosition::new(x.max(0), y.max(0)));
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            open_dashboard,
            hide_dashboard,
            quit_app,
            show_overlay,
            hide_overlay
        ])
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            // ── System tray ──────────────────────────────────────────────
            let open_item =
                MenuItem::with_id(app, "open_dashboard", "Open Dashboard", true, None::<&str>)?;
            let toggle_item = MenuItem::with_id(
                app,
                "toggle_overlay",
                "Show / Hide Overlay",
                true,
                None::<&str>,
            )?;
            let quit_item = MenuItem::with_id(app, "quit", "Quit Jarvis", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&open_item, &toggle_item, &quit_item])?;

            let _tray = TrayIconBuilder::new()
                .icon(app.default_window_icon().unwrap().clone())
                .tooltip("Jarvis")
                .menu(&menu)
                .show_menu_on_left_click(true)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "open_dashboard" => {
                        if let Some(w) = app.get_webview_window("dashboard") {
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
                    }
                    "toggle_overlay" => {
                        if let Some(w) = app.get_webview_window("overlay") {
                            if w.is_visible().unwrap_or(false) {
                                let _ = w.hide();
                            } else {
                                let _ = w.show();
                                let _ = w.set_focus();
                            }
                        }
                    }
                    "quit" => app.exit(0),
                    _ => {}
                })
                .build(app)?;

            // ── Overlay window: position + hide-to-tray on close ─────────
            if let Some(overlay) = app.get_webview_window("overlay") {
                position_overlay(&overlay);
                let overlay_clone = overlay.clone();
                overlay.on_window_event(move |event| {
                    if let WindowEvent::CloseRequested { api, .. } = event {
                        api.prevent_close();
                        let _ = overlay_clone.hide();
                    }
                });
            }

            // ── Dashboard window: hide on close instead of exiting ───────
            if let Some(dashboard) = app.get_webview_window("dashboard") {
                let dashboard_clone = dashboard.clone();
                dashboard.on_window_event(move |event| {
                    if let WindowEvent::CloseRequested { api, .. } = event {
                        api.prevent_close();
                        let _ = dashboard_clone.hide();
                    }
                });
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
