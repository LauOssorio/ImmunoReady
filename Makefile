# can be very usefull wher our preprocess, train, evaluate & predict are built

run preprocess:
	python -c 'from immuno_ready.interface.main import preprocess; preprocess()'

run_trainLSTM:
	python -c 'from immuno_ready.interface.main import train_LSTM; train_LSTM()'

run_pred:
	python -c 'from immuno_ready.interface.main import pred; pred("MKLVAGSEDFRYHQNPTWCI")'

run_api:
	uvicorn immuno_ready.api.fast:app --reload




docker_build_local_dev:
	docker build -t $$GAR_IMAGE:dev .

docker_run_local:
	docker run -it -p 8080:8080 $$GAR_IMAGE:dev

docker_build_local_prod:
	docker build \
		--platform linux/amd64 \
		-t $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/immunoready/$$GAR_IMAGE:prod .

docker_push_prod:
	docker push $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/immunoready/$$GAR_IMAGE:prod

docker_run_GAR:
	gcloud run deploy $$GAR_IMAGE \
		--image $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/immunoready/$$GAR_IMAGE:prod \
		--memory $$GAR_MEMORY \
		--region $$GCP_REGION \
		--env-vars-file .env.yaml \
		--allow-unauthenticated
