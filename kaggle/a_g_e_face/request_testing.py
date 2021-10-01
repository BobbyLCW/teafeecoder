import requests
import base64
from io import BytesIO
from PIL import Image

def sentRequest(imgpath, url):
    img = Image.open(imgpath).convert('RGB')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    base64Img = base64.b64encode(buffer.getvalue())
    base64Img = base64Img.decode('utf-8')

    payload = {'img': base64Img}

    ret = requests.post(url, json=payload)
    content = ret.json()
    print(content)

if __name__ == '__main__':
    image_path = r'C:\Users\mfbob\OneDrive\Desktop\age_gender_eth\train\white_male_39_20170104210052987.jpg'
    urls = 'http://127.0.0.1:9999/age/predict'
    sentRequest(image_path, urls)
