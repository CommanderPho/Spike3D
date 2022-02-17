echo "LaunchInteractiveQt"
%windir%\System32\cmd.exe "/K" C:\Users\pho\miniconda3\Scripts\activate.bat C:\Users\pho\miniconda3\envs\phoviz_ultimate
@REM %windir%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command "& 'C:\Users\pho\miniconda3\shell\condabin\conda-hook.ps1' ; conda activate 'C:\Users\pho\miniconda3' "
%windir%\System32\cmd.exe "/K" C:\Users\pho\miniconda3\Scripts\activate.bat C:\Users\pho\miniconda3

@REM "C:\\Users\\pho\\bin\\PhoWindowsHelpers\\icons\\iPy_Console_Blue_LightRim.ico"

jupyter qtconsole 
@REM change directory to PhoPy3DPositionAnalysis2021 repo
cd 'C:\\Users\\pho\\repos\\PhoPy3DPositionAnalysis2021'

@REM cd c:\Users\pho\repos\PhoPy3DPositionAnalysis2021 
@REM dir

