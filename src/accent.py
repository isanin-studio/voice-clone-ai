"""Подбор способа задать ударение: синтезирует варианты записи слова.

Прогоняет одну и ту же фразу в нескольких нотациях и складывает WAV
в out/accent/. Прослушать и выбрать приём, который работает.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ffmpeg_dlls  # noqa: F401  регистрирует DLL FFmpeg

os.environ.setdefault("COQUI_TOS_AGREED", "1")

import torch
from speak import MODEL, ROOT, resolve_refs
from TTS.api import TTS

ACUTE = "́"  # комбинирующий знак ударения
VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"


def variants(word: str, position: int) -> dict[str, str]:
    """Варианты записи слова с ударением на гласной номер `position`."""
    indexes = [i for i, char in enumerate(word) if char in VOWELS]
    if not 1 <= position <= len(indexes):
        raise SystemExit(f"В слове {len(indexes)} гласных, указана {position}")

    index = indexes[position - 1]
    return {
        "plain": word,
        "acute": word[: index + 1] + ACUTE + word[index + 1 :],
        "upper": word[:index] + word[index].upper() + word[index + 1 :],
        "doubled": word[: index + 1] + word[index] + word[index + 1 :],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("word", help="проблемное слово")
    parser.add_argument("position", type=int, help="номер ударной гласной")
    parser.add_argument("-c", "--context", default="{}",
                        help="фраза с {} на месте слова")
    parser.add_argument("-r", "--ref", default="data/voice")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(MODEL).to(device)
    refs = resolve_refs(args.ref)

    out_dir = ROOT / "out" / "accent"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, spelling in variants(args.word, args.position).items():
        path = out_dir / f"{args.word}_{name}.wav"
        tts.tts_to_file(
            text=args.context.format(spelling),
            speaker_wav=refs,
            language="ru",
            file_path=str(path),
        )
        print(f"{name:8} {spelling:20} -> {path.name}")

    print(f"\nПрослушать: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
