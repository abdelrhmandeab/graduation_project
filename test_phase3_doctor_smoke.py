from core.doctor import _probe_ollama_models, _probe_vram_status


def main():
    models_ok, models_details = _probe_ollama_models()
    vram_ok, vram_details = _probe_vram_status()

    print("models_ok:", models_ok)
    print("models_details:", models_details)
    print("vram_ok:", vram_ok)
    print("vram_details:", vram_details)

    assert isinstance(models_ok, bool)
    assert isinstance(vram_ok, bool)
    assert isinstance(models_details, str)
    assert isinstance(vram_details, str)


if __name__ == "__main__":
    main()
