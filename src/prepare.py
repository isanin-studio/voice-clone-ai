"""Конвертация любых аудиофайлов в референсы XTTS: 22.05 кГц, моно.

Обрезает тишину по краям, нормализует пик, отбраковывает слишком
короткие записи.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ffmpeg_dlls  # noqa: F401  регистрирует DLL FFmpeg

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 22050
MIN_SECONDS = 3.0
TRIM_DB = 30  # порог тишины относительно пика


def convert(src: Path, dst: Path) -> bool:
    audio, _ = librosa.load(src, sr=SAMPLE_RATE, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=TRIM_DB)

    duration = len(audio) / SAMPLE_RATE
    if duration < MIN_SECONDS:
        print(f"ПРОПУСК {src.name}: всего {duration:.1f} c")
        return False

    peak = float(np.max(np.abs(audio))) or 1.0
    audio = audio / peak * 0.95

    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, audio, SAMPLE_RATE)
    print(f"{src.name} -> {dst.name}  {duration:.1f} c, пик {peak:.2f}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", help="файлы или папки")
    parser.add_argument("-o", "--out-dir", default="data/voice")
    parser.add_argument("-p", "--prefix", default="ref")
    args = parser.parse_args()

    files: list[Path] = []
    for item in args.sources:
        path = Path(item)
        if path.is_dir():
            files += sorted(p for p in path.iterdir() if p.is_file())
        else:
            files += sorted(path.parent.glob(path.name))

    if not files:
        raise SystemExit("Файлы не найдены.")

    out_dir = Path(args.out_dir)
    index = 1
    for src in files:
        dst = out_dir / f"{args.prefix}{index}.wav"
        try:
            if convert(src, dst):
                index += 1
        except Exception as error:  # формат может не читаться
            print(f"ОШИБКА {src.name}: {error}")

    print(f"Готово: {index - 1} референсов в {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
