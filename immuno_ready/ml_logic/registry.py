import time
from google.cloud import storage
from immuno_ready.params import *
from colorama import Fore, Style

def save_model(model):
    """
    Persist trained model it in your bucket on GCS at "models/{timestamp}.h5"
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"{timestamp}.h5"

    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models","immunoReady", f"{timestamp}.h5")
    model.save(model_path)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(model_path)

    print("✅ Model saved locally and to GCS")

    return None

def load_model():
    """
    Return latest saved model from GCS
    """
    print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)
    client = storage.Client()
    blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

    try:
        latest_blob = max(blobs, key=lambda x: x.updated)
        latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
        latest_blob.download_to_filename(latest_model_path_to_save)

        latest_model = keras.models.load_model(latest_model_path_to_save)

        print("✅ Latest model downloaded from cloud storage")

        return latest_model
    except:
        print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

        return None
