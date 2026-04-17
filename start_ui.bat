@echo off
echo Starting User Interface...
cd /d "C:\Users\Anass\Desktop\projet3.0"
set PYTHONPATH=C:\Users\Anass\Desktop\projet3.0
streamlit run src/ui/streamlit_app.py --server.port 8501
pause
