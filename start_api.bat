@echo off
echo Starting Recommendation API...
cd /d "C:\Users\Anass\Desktop\projet3.0"
set PYTHONPATH=C:\Users\Anass\Desktop\projet3.0
python -m uvicorn src.serving.recommendation_api:app --host 0.0.0.0 --port 8000 --reload
pause
