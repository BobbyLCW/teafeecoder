import uvicorn
from fastapi import FastAPI
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
import base64
from AgeGenderObject import AGEObject

app = FastAPI()
device = 'cpu'
model_path = 'age_gender_best.pth'
ethMap = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}
genderMap = {0: 'male', 1: 'female'}
model = torch.load(model_path, map_location=device)

def decodeImg(imgstr):
    b64img = base64.b64decode(imgstr)
    buffer = BytesIO(b64img)
    return Image.open(buffer)

@app.get('/')
def hello_world():
    return "Welcome to Age, Gender, Ethnicity Predictions!"

@app.post('/age/predict')
def inference(data: AGEObject):
    data = data.dict()
    mean = torch.tensor([0.5], dtype=torch.float32)
    std = torch.tensor([0.5], dtype=torch.float32)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),])

    imgstr = data['img']
    img = decodeImg(imgstr)
    img = transform(img)

    forward = model(img.unsqueeze(1)) # convert (1,48,48) to (1,1,48,48)
    eth = int(forward['ethnicity'].argmax(1).tolist()[0])
    gender = int(forward['gender'].squeeze(1).tolist()[0])
    age = round(forward['age'].squeeze(1).tolist()[0])

    resp = {'ethnicity': ethMap[eth], 'gender': genderMap[gender], 'age': age}

    return resp

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=9999)