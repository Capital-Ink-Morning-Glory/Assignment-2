import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from lib.model import CNN
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image


def predict_img(model, img_path: str):
    """
    :param model: The trained model.
    :param img_path: The path of an image.
    :return: Which label is most likely to be.
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = read_image(img_path, mode=ImageReadMode.RGB)
    img = transform(img)
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    preds_single = [classes[idx] for idx in predicted.tolist()]  # Index to class label in string
    return preds_single


trained_model = CNN()
trained_model.load_state_dict(torch.load('./model/CNN.pth', weights_only=True))

app = FastAPI()
app.mount("/app/static", StaticFiles(directory="./static"), name="static")  # mount static directory

classes = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship",
           9: "truck"}


@app.get("/")
def root():
    return {"message": "Image Classification FastAPI"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    img_path = f"./static/{file.filename}"
    with open(img_path, "wb") as f:
        f.write(await file.read())

    try:
        label = predict_img(trained_model, img_path)
        return JSONResponse({
            "filename": file.filename,
            "prediction": label
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
