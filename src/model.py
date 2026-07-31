from tensorflow.keras.layers import *

from tensorflow.keras.models import Model

from config import IMG_SIZE


def attention_block(x):

    attention=GlobalAveragePooling2D()(x)

    attention=Dense(
        x.shape[-1]//8,
        activation="relu"
    )(attention)

    attention=Dense(
        x.shape[-1],
        activation="sigmoid"
    )(attention)

    attention=Reshape((1,1,x.shape[-1]))(attention)

    x=Multiply()([x,attention])

    return x


def build_model():

    input_layer=Input(
        shape=(IMG_SIZE,IMG_SIZE,1)
    )

    x=Conv2D(32,(3,3),activation="relu")(input_layer)
    x=MaxPooling2D()(x)

    x=Conv2D(64,(3,3),activation="relu")(x)
    x=MaxPooling2D()(x)

    x=Conv2D(128,(3,3),activation="relu")(x)
    x=MaxPooling2D()(x)

    x=attention_block(x)

    x=Flatten()(x)

    x=Dense(128,activation="relu")(x)

    x=Dropout(0.5)(x)

    output=Dense(3,activation="softmax")(x)

    model=Model(input_layer,output)

    return model