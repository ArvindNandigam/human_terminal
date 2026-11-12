@echo off
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
echo Setup complete!
echo Run: python -m hcmd "list files"
pause
