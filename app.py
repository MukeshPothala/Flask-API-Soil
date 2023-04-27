from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import cv2    
from PIL import Image
import os
import base64
# from flask_cors import CORS,cross_origin

Pmodel = pickle.load(open('Pclassifier.pkl', 'rb'))
pHmodel = pickle.load(open('pHclassifier.pkl', 'rb'))
OMmodel = pickle.load(open('OMclassifier.pkl', 'rb'))
ECmodel = pickle.load(open('ECclassifier.pkl', 'rb'))

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# CORS(app,supports_credentials=True)

@app.route('/')
# @cross_origin(origins='*')

def home():
    return "Serving service to mobile application with api to respond to image with analysis"
    # return render_template("form.html")

@app.route('/predict', methods = ['POST', "GET"])
# @cross_origin(origins='*')
def predict():
    if request.method == "POST":
        # inputImage = request.form.get('inputImage')
        f = request.files['inputImage'].read()
        #    print(f)
        npimg = np.fromstring(f,np.uint8)
        image = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
        # img = Image.fromarray(img.astype("uint8"))
        # f = request.files['inputImage']
        # filename = secure_filename(f.filename)
        # print(filename)
        # f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # with open(filename, "rb") as img_file:
        #     my_string = base64.b64encode(img_file.read())
        # print(my_string)
        # nparr = np.frombuffer(base64.b64decode(), np.uint8)
        # image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        print(image.shape)
        # extracting blue,red,green channel from color image
        blue_channel = image[:,:,0]
        green_channel = image[:,:,1]
        red_channel = image[:,:,2]
        temp = ((np.median(green_channel)+np.median(blue_channel))+np.median(red_channel))
        temp = np.nanmean(temp)
        Presult = float(Pmodel.predict([[temp]]))
        pHreuslt = float(pHmodel.predict([[temp]]))
        OMresult = float(OMmodel.predict([[temp]]))
        ECresult = float(ECmodel.predict([[temp]]))
        
        result = {'P':Presult, 'pH':pHreuslt, 'OM':OMresult, 'EC':ECresult } 
        print(result)
        return jsonify(result)
    # else:
    #     return "API is running to handle android application requests"
    return render_template("form.html")

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0.0')
    

# #Ecoding image to base64
# with open("testing.jpg", "rb") as img_file:
#     b64_string = base64.b64encode(img_file.read())
# print(b64_string)
# with open("q1.txt","w+") as f:
#     f.write(b64_string.decode("utf-8"))



