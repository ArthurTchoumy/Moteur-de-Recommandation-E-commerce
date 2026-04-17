@echo off
echo Starting Admin Dashboard...
cd /d "C:\Users\Anass\Desktop\projet3.0"
set PYTHONPATH=C:\Users\Anass\Desktop\projet3.0
streamlit run src/ui/admin_dashboard.py --server.port 8502
pause
