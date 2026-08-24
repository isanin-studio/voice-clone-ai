"""Регистрация каталога DLL FFmpeg для torchcodec (Windows).

torchcodec грузит libtorchcodec_core*.dll, которая тянет avcodec/avutil.
Python 3.8+ не ищет зависимости расширений в PATH, поэтому каталог
добавляется явно. Импортировать ДО torch/TTS.

Путь берётся из переменной FFMPEG_BIN, иначе ищется в пакетах winget.
"""
import os
import sys
from pathlib import Path

WINGET_PACKAGES = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages"


def find_ffmpeg_bin() -> Path | None:
    explicit = os.environ.get("FFMPEG_BIN")
    if explicit:
        return Path(explicit)

    if not WINGET_PACKAGES.is_dir():
        return None

    # torchcodec 0.16 умеет FFmpeg 4..7 — свежие сборки не подходят
    candidates = sorted(WINGET_PACKAGES.glob("*FFmpeg*Shared*/*/bin/avcodec-6?.dll"))
    return candidates[-1].parent if candidates else None


def register() -> Path | None:
    if sys.platform != "win32":
        return None

    bin_dir = find_ffmpeg_bin()
    if not bin_dir or not bin_dir.is_dir():
        return None

    os.add_dll_directory(str(bin_dir))
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    return bin_dir


register()
