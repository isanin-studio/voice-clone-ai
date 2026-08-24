"""Запись референсного образца голоса с микрофона в 22.05 кГц моно."""
import argparse
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 22050
ROOT = Path(__file__).resolve().parents[1]


def list_devices() -> None:
    print(sd.query_devices())


def record(seconds: float, out_path: Path, device: int | None) -> None:
    print(f"Запись {seconds:.0f} c. Говорите ровно, без пауз и фонового шума.")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()

    peak = float(np.max(np.abs(audio))) or 1.0
    audio = audio / peak * 0.95  # нормализация пика

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, SAMPLE_RATE)
    print(f"Сохранено: {out_path}  (пик исходника {peak:.3f})")
    if peak < 0.05:
        print("ВНИМАНИЕ: сигнал слишком тихий, проверьте микрофон.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", default="data/voice/ref.wav")
    parser.add_argument("-s", "--seconds", type=float, default=20.0)
    parser.add_argument("-d", "--device", type=int, default=None)
    parser.add_argument("--list", action="store_true", help="показать устройства")
    args = parser.parse_args()

    if args.list:
        list_devices()
        return 0

    record(args.seconds, ROOT / args.out, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
