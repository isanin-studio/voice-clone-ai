# Озвучить содержимое data/text.txt и воспроизвести результат.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

& .\.venv\Scripts\python.exe src\speak.py -f data/text.txt -o out/speech.wav
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

(New-Object Media.SoundPlayer "$PSScriptRoot\out\speech.wav").PlaySync()
