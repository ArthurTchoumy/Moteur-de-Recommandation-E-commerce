@echo off
echo Starting Simple User Interface...
cd /d "%~dp0"
set PYTHONPATH=%~dp0
streamlit run src/ui/simple_app.py --server.port 8501
pause
