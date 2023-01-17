from fastapi import FastAPI

app = FastAPI()


from http import HTTPStatus


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}


from enum import Enum


class ItemEnum(Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


@app.get("/restric_items/{item_id}")
def read_item(item_id: ItemEnum):
    return {"item_id": item_id}


@app.get("/query_items")
def read_item(item_id: int):
    return {"item_id": item_id}


from fastapi import UploadFile, File
from typing import Optional

import cv2
from fastapi.responses import FileResponse


@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...), h: int = 28, w: int = 28):
    # with open("image.jpg", "wb") as image:
    #    content = await data.read()
    #    image.write(content)
    img = cv2.imread("image.jpg")
    res = cv2.resize(img, (h, w))
    #    image.close()

    cv2.imwrite("image_resize.jpg", res)
    return FileResponse("image_resize.jpg")
