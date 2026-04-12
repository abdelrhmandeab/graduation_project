# User Guide

## 1. What Jarvis Can Do

Jarvis is a local Windows desktop assistant for English and Arabic.

Core capabilities:
- Open and close applications.
- Search and manage files and folders.
- Execute safe system commands.
- Answer natural language questions.
- Continue multi-turn follow-up commands.

Supported languages:
- English
- Arabic

## 2. Quick Start

1. Install Python 3.12+ on Windows.
2. Open PowerShell in the project root.
3. Run setup:

```powershell
./scripts/setup_windows.ps1
```

4. Start Ollama in a separate terminal:

```powershell
ollama serve
```

5. Start Jarvis:

```powershell
python core/orchestrator.py
```

If audio dependencies are missing, Jarvis falls back to text interaction.

## 3. Everyday Commands

### English
- open notepad
- close calculator
- find file report in desktop
- go to downloads
- list files
- voice status
- observability

### Arabic
- افتح المفكرة
- اقفل الحاسبة
- دور على ملف تقرير في المكتب
- روح على التنزيلات
- وريني الملفات
- حالة الصوت

## 4. Follow-up Commands

You can use context-aware commands:
- open it
- delete it
- rename it to final_report.txt
- confirm it

Arabic follow-up examples:
- افتحه
- امسحه
- غيره ل التقرير_النهائي.txt
- اكد

## 5. Safety and Confirmation

Risky actions require confirmation tokens.

Example flow:
1. You request delete.
2. Jarvis returns a token.
3. You run:

```text
confirm <token> <PIN_or_passphrase>
```

If too many wrong second-factor attempts occur, the token is temporarily locked.

## 6. Voice Profiles and Audio UX

Useful commands:
- voice diagnostic
- voice quality natural
- voice quality standard
- stt profile quiet
- stt profile noisy
- audio ux profile balanced
- audio ux profile responsive
- audio ux profile robust

## 7. Observability and Health

Use these commands for runtime visibility:
- observability
- memory status
- policy status
- verify audit log

For environment diagnostics:

```powershell
python core/doctor.py
```

## 8. Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and fixes.
