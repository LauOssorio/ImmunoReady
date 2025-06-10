import time
import pickle
from google.cloud import storage
from tensorflow import keras
from colorama import Fore, Style
from immuno_ready.params import *


def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", "immunoready", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", "immunoready", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")

def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model it in your bucket on GCS at "models/immunoready/{timestamp}.keras"
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"{timestamp}.keras"

    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", "immunoready", f"{timestamp}.keras")
    model.save(model_path)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(model_path)

    print("✅ Model saved locally and to GCS")

    return None

def load_model() -> keras.Model:
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
