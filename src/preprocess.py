import cv2
import numpy as np
import os

from tensorflow.keras.utils import to_categorical

from config import IMG_SIZE, CATEGORIES


def apply_clahe(image):
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )
    return clahe.apply(image)


def adjust_brightness(image):
    factor = np.random.uniform(0.8,1.2)
    return np.clip(image*factor,0,255).astype(np.uint8)


def load_dataset(dataset_path):

    data=[]
    labels=[]

    for category in CATEGORIES:

        path=os.path.join(dataset_path,category)

        label=CATEGORIES.index(category)

        for img in os.listdir(path):

            try:

                image=cv2.imread(
                    os.path.join(path,img),
                    cv2.IMREAD_GRAYSCALE
                )

                image=cv2.resize(
                    image,
                    (IMG_SIZE,IMG_SIZE)
                )

                image=apply_clahe(image)

                image=adjust_brightness(image)

                image=image/255.0

                data.append(image)

                labels.append(label)

            except:
                pass

    data=np.array(data).reshape(-1,IMG_SIZE,IMG_SIZE,1)

    labels=to_categorical(labels,3)

    return data,labels