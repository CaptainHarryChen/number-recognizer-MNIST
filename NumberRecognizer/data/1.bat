@echo off

setlocal enabledelayedexpansion

set "ext=%~x1"

:loop

if defined ext set "ext=!ext:"=!"

if defined ext goto ok

echo ����㲻֪���ļ�����չ�����ر�������Ȼ����ļ��ϵ��������ļ���ͼ���ϡ�

set /p "v=��������չ������txt��Ȼ��س���"

for /f "delims=" %%i in (".!v!") do set "ext=%%~xi"

goto loop

:ok

echo ��չ����!ext!

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