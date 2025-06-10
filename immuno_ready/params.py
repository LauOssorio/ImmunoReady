import os

##################  VARIABLES  ##################
GCP_REGION = os.environ.get("GCP_REGION")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
GAR_IMAGE = os.environ.get("GAR_IMAGE")
GAR_MEMORY = os.environ.get("GAR_MEMORY")
PORT= os.environ.get("PORT")
PCA_NB= os.environ.get("PCA_NB")

##################  CONSTANTS  #####################
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")

################## VALIDATIONS #################
