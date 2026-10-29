.PHONY: env gpu download unify train evaluate train-anomaly api demo test

env:
	conda env create -f environment.yml

gpu:
	python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

download:
	python -m src.data.download

unify:
	python -m src.data.unify

train:
	python -m src.supervised.train

evaluate:
	python -m src.supervised.evaluate

train-anomaly:
	python -m src.unsupervised.train_anomaly

api:
	uvicorn api.main:app --reload

demo:
	python demo/app.py

test:
	pytest
