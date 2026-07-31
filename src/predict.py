import cv2
import numpy as np

from tensorflow.keras.models import load_model

from preprocess import apply_clahe

from config import *


model=load_model(MODEL_PATH)


def predict_image(img_path):

    image=cv2.imread(
        img_path,
        cv2.IMREAD_GRAYSCALE
    )

    image=cv2.resize(
        image,
        (IMG_SIZE,IMG_SIZE)
    )

    image=apply_clahe(image)

    image=image/255.0

    image=image.reshape(
        1,
        IMG_SIZE,
        IMG_SIZE,
        1
    )

    prediction=model.predict(image)

    index=np.argmax(prediction)

    return CATEGORIES[index]