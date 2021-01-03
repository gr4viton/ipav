echo Current working directory is %cd%
cd %~dp0
echo Changing direcotory to %cd%

python.exe -c "print('\7')" :: BEEP

python.exe main.py
