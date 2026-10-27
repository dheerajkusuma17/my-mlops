setup:
	python -m venv venv
	venv\Scripts\pip install --upgrade pip
	venv\Scripts\pip install -r requirements.txt

train:
	python train.py

serve:
	uvicorn app:app --reload

mlflow:
	mlflow ui

test:
	python -m pytest

clean:
	if exist __pycache__ rmdir /s /q __pycache__
	if exist mlruns rmdir /s /q mlruns
 