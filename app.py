import io
from io import BytesIO
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torchvision.transforms as transforms
from models import CNN
import matplotlib.pyplot as plt
import re


app = Flask(__name__)

model = CNN(input_shape=1, output_shape=10)
model.load_state_dict(torch.load("MNIST_cnn.pth", weights_only=True),)
model.eval()

transforms = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((28,28)),
        transforms.ToTensor()
    ]
)

CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json['image']

    image_data = base64.b64decode(data.split(',')[1])
    image = Image.open(BytesIO(image_data))

    img = transforms(image).unsqueeze(0)

    # plt.imshow(img[0].permute(1, 2, 0), cmap="gray")
    # plt.show()

    with torch.no_grad():
        output = model(img)
        pred = output.argmax(dim=1, keepdim=True).item()

    return jsonify({"prediction":pred})

app.run(debug=True)
