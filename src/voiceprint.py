"""Слепок голоса: извлечение и синтез по нему.

Слепок (conditioning latents + speaker embedding) сохраняется в
data/voice/voiceprint.pth — около 100 КБ. Синтез по слепку не требует
референсных WAV и работает быстрее.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ffmpeg_dlls  # noqa: F401  регистрирует DLL FFmpeg
from accents import apply_stress

os.environ.setdefault("COQUI_TOS_AGREED", "1")

import torch
from speak import MODEL, ROOT, resolve_refs

PRINT_PATH = ROOT / "data" / "voice" / "voiceprint.pth"
OUTPUT_RATE = 24000


def load_model():
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")
    return TTS(MODEL).to(device).synthesizer.tts_model


def extract(ref_pattern: str) -> None:
    model = load_model()
    refs = resolve_refs(ref_pattern)
    print(f"Референсы: {len(refs)}")

    gpt_latent, embedding = model.get_conditioning_latents(audio_path=refs)
    PRINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"gpt_cond_latent": gpt_latent.cpu(), "speaker_embedding": embedding.cpu()},
        PRINT_PATH,
    )
    size_kb = PRINT_PATH.stat().st_size / 1024
    print(f"Слепок сохранён: {PRINT_PATH}  ({size_kb:.0f} КБ)")


def synthesize(text: str, out_path: Path) -> None:
    if not PRINT_PATH.exists():
        raise SystemExit(f"Слепок не найден. Сначала: voiceprint.py extract")

    import soundfile as sf

    model = load_model()
    device = next(model.parameters()).device
    data = torch.load(PRINT_PATH, weights_only=False)

    result = model.inference(
        text,
        "ru",
        data["gpt_cond_latent"].to(device),
        data["speaker_embedding"].to(device),
        enable_text_splitting=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, result["wav"], OUTPUT_RATE)
    print(f"Готово: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    extract_cmd = sub.add_parser("extract", help="создать слепок из референсов")
    extract_cmd.add_argument("-r", "--ref", default="data/voice")

    say_cmd = sub.add_parser("say", help="синтез по слепку")
    say_cmd.add_argument("text", nargs="?")
    say_cmd.add_argument("-f", "--file")
    say_cmd.add_argument("-o", "--out", default="out/speech.wav")

    args = parser.parse_args()

    if args.command == "extract":
        extract(args.ref)
        return 0

    text = (ROOT / args.file).read_text(encoding="utf-8") if args.file else args.text
    if not text or not text.strip():
        raise SystemExit("Пустой текст.")
    synthesize(apply_stress(text.strip()), ROOT / args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
