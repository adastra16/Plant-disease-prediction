import os
import json
from PIL import Image

import numpy as np

# Suppress verbose TensorFlow logs and oneDNN notice before importing TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

import tensorflow as tf
import streamlit as st


working_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(working_dir, "trained_model")
model_path = os.path.join(model_dir, "plant_disease_prediction_model.h5")


def _download_from_google_drive_with_requests(file_id: str, destination: str) -> None:
    """Downloads a Google Drive file using requests, handling the confirmation token for large files."""
    import requests  # local import to avoid requiring it unless needed

    def get_confirm_token(resp):
        for key, value in resp.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    def save_response_content(resp, dest):
        chunk_size = 32768
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)

    session = requests.Session()
    URL = "https://drive.google.com/uc?export=download"

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def _download_from_generic_url(url: str, destination: str) -> None:
    """Downloads a file from a direct HTTP(S) URL using requests."""
    import requests

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def _looks_like_hdf5(path: str) -> bool:
    """Quick check whether file begins with the HDF5 magic signature."""
    try:
        with open(path, 'rb') as f:
            sig = f.read(8)
        return sig == b"\x89HDF\r\n\x1a\n"
    except Exception:
        return False


def _extract_url_and_file_id(link_line: str):
    """Return (url, drive_file_id|None). Accepts a full line that may include prefix text."""
    import re
    # Find the first URL in the line if present
    url_match = re.search(r"https?://\S+", link_line)
    url = url_match.group(0) if url_match else link_line.strip()

    file_id = None
    if "/d/" in url:
        try:
            file_id = url.split("/d/")[1].split("/")[0]
        except Exception:
            file_id = None
    elif "id=" in url:
        try:
            file_id = url.split("id=")[1].split("&")[0]
        except Exception:
            file_id = None
    # If the whole content looks like an id string
    if file_id is None and len(url) > 20 and "/" not in url and "http" not in url:
        file_id = url
    return url, file_id


# Ensure the trained model exists; if not, try to download using the provided link
if not os.path.exists(model_path):
    link_file_path = os.path.join(model_dir, "trained_model_link.txt")
    if os.path.exists(link_file_path):
        try:
            with open(link_file_path, "r", encoding="utf-8") as f:
                link_line = f.read().strip()
            url, file_id = _extract_url_and_file_id(link_line)

            if file_id or (url and url.startswith("http")):
                try:
                    # Prefer gdown for Drive IDs; fallback to requests
                    if file_id:
                        try:
                            import gdown  # type: ignore
                            os.makedirs(model_dir, exist_ok=True)
                            gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
                        except Exception:
                            _download_from_google_drive_with_requests(file_id, model_path)
                    else:
                        _download_from_generic_url(url, model_path)
                except Exception as download_error:
                    st.warning(f"Could not auto-download model: {download_error}")

                # Validate downloaded file; if not a proper HDF5, remove and warn
                if os.path.exists(model_path) and not _looks_like_hdf5(model_path):
                    try:
                        os.remove(model_path)
                    except Exception:
                        pass
                    st.error("Downloaded file does not look like a valid Keras .h5 model (received HTML or an error page). Please download manually.")
                    if url:
                        st.link_button("Open model link", url)
            else:
                st.warning("Could not find a valid URL or Google Drive ID in trained_model_link.txt. Please download the model manually.")
        except Exception as e:
            st.warning(f"Failed reading trained_model_link.txt: {e}")
    else:
        st.info("Model file is missing and no download link was found. Place 'plant_disease_prediction_model.h5' into 'app/trained_model/'.")

# Try loading the model if it now exists
model = None
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        # Provide a quick action to open the link file
        link_file_path = os.path.join(model_dir, "trained_model_link.txt")
        if os.path.exists(link_file_path):
            try:
                with open(link_file_path, "r", encoding="utf-8") as f:
                    link_line = f.read().strip()
                url, _ = _extract_url_and_file_id(link_line)
                if url and url.startswith("http"):
                    st.link_button("Open model link", url)
            except Exception:
                pass
else:
    st.warning("Model file not found. Please ensure the model is available at 'app/trained_model/plant_disease_prediction_model.h5'.")


class_indices = json.load(open(f"{working_dir}/class_indices.json"))


def load_and_preprocess_image(image_path, target_size=(224, 224)):
 
    img = Image.open(image_path)

    img = img.resize(target_size)

    img_array = np.array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array.astype('float32') / 255.
    return img_array


def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name



st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            if model is None:
                st.error("Model not loaded. Please ensure the model file is present and reload the app.")
            else:
                # Reset the file pointer as it may have been consumed when previewing
                try:
                    uploaded_image.seek(0)
                except Exception:
                    pass
                # Preprocess the uploaded image and predict the class
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f'Prediction: {str(prediction)}')
