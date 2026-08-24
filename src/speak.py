"""Синтез речи вашим голосом (XTTS-v2, zero-shot клонирование)."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ffmpeg_dlls  # noqa: F401  регистрирует DLL FFmpeg
from accents import apply_stress

os.environ.setdefault("COQUI_TOS_AGREED", "1")

import torch
from TTS.api import TTS

# torch>=2.6 грузит чекпойнты с weights_only=True и падает на классах XTTS
_orig_load = torch.load


def _load_full(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)


torch.load = _load_full

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
ROOT = Path(__file__).resolve().parents[1]


def resolve_refs(pattern: str) -> list[str]:
    path = ROOT / pattern
    if path.is_dir():
        refs = sorted(path.glob("*.wav"))
    else:
        refs = sorted(path.parent.glob(path.name))
    if not refs:
        raise SystemExit(f"Референсы не найдены: {pattern}")
    return [str(p) for p in refs]


def read_text(args: argparse.Namespace) -> str:
    if args.file:
        raw = (ROOT / args.file).read_text(encoding="utf-8")
    elif args.text:
        raw = args.text
    else:
        raw = sys.stdin.read()
    return apply_stress(raw.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="текст для озвучки")
    parser.add_argument("-f", "--file", help="файл с текстом (UTF-8)")
    parser.add_argument("-r", "--ref", default="data/voice", help="wav или папка")
    parser.add_argument("-o", "--out", default="out/speech.wav")
    parser.add_argument("-l", "--lang", default="ru")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    text = read_text(args)
    if not text:
        raise SystemExit("Пустой текст.")

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Устройство: {device}")

    tts = TTS(MODEL).to(device)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tts.tts_to_file(
        text=text,
        speaker_wav=resolve_refs(args.ref),
        language=args.lang,
        speed=args.speed,
        file_path=str(out_path),
        split_sentences=True,
    )
    print(f"Готово: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
