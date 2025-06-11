# can be very usefull wher our preprocess, train, evaluate & predict are built

run preprocess:
	python -c 'from immuno_ready.interface.main import preprocess; preprocess()'

run_trainLSTM:
	python -c 'from immuno_ready.interface.main import train_LSTM; train_LSTM()'

run_pred:
	python -c 'from immuno_ready.interface.main import pred; pred("MKLVAGSEDFRYHQNPTWCI")'

run_api:
	uvicorn immuno_ready.api.fast:app --reload





docker run -it
docker run -it -e PORT=8000 
