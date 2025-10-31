@echo off
setlocal

rem Processa tutti i file che iniziano con "hando" e terminano con .mp4 nella directory attuale
for %%f in (hando*.mp4) do (
    rem Crea un file temporaneo per il resize
    ffmpeg -i "%%f" -vf scale=720:1280 -y "%%~dpn_temp.mp4"

    rem Ruota a sinistra di 90° il file ridimensionato e sovrascrive l'originale
    ffmpeg -i "%%~dpn_temp.mp4" -vf "transpose=2" -y "%%f"

    rem Elimina il file temporaneo
    del "%%~dpn_temp.mp4"
)

endlocal
pause
