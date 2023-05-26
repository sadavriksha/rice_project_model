from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
# import opencv as cv
app = FastAPI()

# D:\rice-disease\saved_models\2
MODEL = tf.keras.models.load_model("./model/4")

CLASS_NAMES = ['Bacterial_leaf_blight', 'Bacterial_leaf_streak', 'Bacterial_panicle_blight', 'Blast', 'Brown_spot',

               'Dead_heart', 'Downy_mildew', 'False_smut', 'Hispa', 'Leaf_scald', 'Neck_blast', 'Normal', 'Tungro']


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    image = np.array(Image.open(BytesIO(data)).resize((256, 256)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):

    # image = await file.read()
    # image = image.resize((256, 256))

    image = read_file_as_image(await file.read())

    print(type(image))
    print(image.shape)
    print(image.size)
    # image = np.reshape(image, (2560, 2566))
    # print(image.shape)
    # img_batch = np.expand_dims(image, 0)

    # image = tf.reshape(image, (-1, 256, 256, 3, 1))
    # image = tf.squeeze(image)

    img_batch = np.expand_dims(image, 0)
    print(type(img_batch))
    print(img_batch.shape)
    print(img_batch.size)
    # image = np.reshape(img_batch, 256)
    # img_batch = img_batch[:65536].reshape((256,256))

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    print(predicted_class,confidence)

    return {
        "class": predicted_class , "confidence": float(confidence)
    }



if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=5000)
