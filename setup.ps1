# Установка окружения. Требуется Python 3.11 или 3.12.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

# torchcodec грузит DLL FFmpeg; статические сборки не подходят,
# версия 8+ им не поддерживается — нужна именно shared 7.1
winget install --id BtbN.FFmpeg.GPL.Shared.7.1 -e `
    --accept-source-agreements --accept-package-agreements

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# RTX 5070 Ti = Blackwell (sm_120): только cu128, иначе "no kernel image"
.\.venv\Scripts\pip.exe install torch torchaudio `
    --index-url https://download.pytorch.org/whl/cu128

.\.venv\Scripts\pip.exe install -r requirements.txt

.\.venv\Scripts\python.exe -c @'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
'@
