import os
import base64
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
from io import BytesIO

# Load disease and supplement info
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load the AI Model
model = CNN.CNN(39)    
model.load_state_dict(torch.load("model.pt"))
model.eval()

# Define upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def prediction(image_path):
    """Function to process image and predict disease index."""
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    print(output)
    index = np.argmax(output)
    print(index)
    print(output[0][index])
    print(disease_info['disease_name'][index])
    return index if output[0][index]>11 and output[0][index]<31 else -1
    # return index

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['POST'])
def submit():
    """Handles both file uploads and camera-captured images."""
    try:
        filename = None

        # Check if image_data is provided (camera capture via Base64)
        if 'image_data' in request.form and request.form['image_data'].strip() != "":
            image_data = request.form['image_data']
            # Verify that image_data is in expected format: "data:image/jpeg;base64,..."
            if not image_data.startswith("data:image"):
                return {"error": "Invalid image data format"}, 400
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(BytesIO(image_bytes))
            filename = os.path.join(app.config['UPLOAD_FOLDER'], "captured_image.jpg")
            image.save(filename)
            print("Captured image saved at:", filename)
        # Else, check for file uploads. Use getlist to handle multiple files with the same key.
        else:
            files = request.files.getlist('image')
            file = None
            for f in files:
                if f.filename.strip() != "":
                    file = f
                    break
            if file is None:
                return {"error": "No image provided"}, 400
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            print("Uploaded file saved at:", filename)

        pred = prediction(filename)
        if pred==-1:
            print("wrong")
            pred=4
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template(
            'submit.html',
            title=title,
            desc=description,
            prevent=prevent,
            image_url=image_url,
            pred=pred,
            sname=supplement_name,
            simage=supplement_image_url,
            buy_link=supplement_buy_link
        )
    except Exception as e:
        print(f"error: {e}")
        pred=4
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template(
            'submit.html',
            title=title,
            desc=description,
            prevent=prevent,
            image_url=image_url,
            pred=pred,
            sname=supplement_name,
            simage=supplement_image_url,
            buy_link=supplement_buy_link
        )



@app.route('/market', methods=['GET'])
def market():
    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link'])
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
