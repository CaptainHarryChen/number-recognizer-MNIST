@echo off

setlocal enabledelayedexpansion

set "ext=%~x1"

:loop

if defined ext set "ext=!ext:"=!"

if defined ext goto ok

echo 如果你不知道文件的扩展名，关闭批处理然后把文件拖到批处理文件的图标上。

set /p "v=请输入扩展名（如txt）然后回车："

for /f "delims=" %%i in (".!v!") do set "ext=%%~xi"

goto loop

:ok

echo 扩展名：!ext!

pause

reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\FileExts\!ext!" /f

reg query "HKCR\!ext!" /ve|find /i "!ext:~1!_auto_file">nul

if not errorlevel 1 (

reg delete "HKCR\!ext!" /ve /f

reg delete "HKCR\!ext:~1!_auto_file" /f

)

taskkill /im explorer.exe /f

start %windir%\explorer.exe

pause

goto :eo