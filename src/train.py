from sklearn.model_selection import train_test_split

from preprocess import load_dataset

from model import build_model

from config import *


data,labels=load_dataset(DATASET_PATH)

X_train,X_test,y_train,y_test=train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42
)

model=build_model()

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history=model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test,y_test)
)

loss,accuracy=model.evaluate(
    X_test,
    y_test
)

print("Accuracy:",accuracy)

model.save(MODEL_PATH)