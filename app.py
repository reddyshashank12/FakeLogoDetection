from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import sqlite3
import pandas as pd
import numpy as np
import pickle
import sqlite3
import random

import smtplib 
from email.message import EmailMessage
from datetime import datetime

from PIL import Image
import seam_carving
from PIL import Image, ImageFilter
import io
import cv2
import numpy as np
from torchvision.models import detection
import sqlite3
import torch
from torchvision import models
from flask import Flask, render_template, request, redirect, Response

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath



app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

# allow files of a specific type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# function to check the file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model_path2 = 'model.h5' # load .h5 Model

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



#classes2 = {0:"APHIDS",1:"ARMYWORM",2:"BEETLE",3:"BOLLWORM",4:"GRASSHOPPER",5:"MITES",6:"MOSQUITO",7:"SAWFLY",8:"STEM BORER"}
model1 = load_model(model_path2, custom_objects={'f1_score' : f1_m, 'precision_score' : precision_m, 'recall_score' : recall_m}, compile=False)

model = torch.hub.load("ultralytics/yolov5", "custom", path = "best.pt", force_reload=True)

model.eval()
model.conf = 0.5  
model.iou = 0.45  

from io import BytesIO

import numpy as np
np.object = np.object_  

from keras.preprocessing.image import load_img, img_to_array

   
   
@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')


@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route('/index1')
def index1():
	return render_template('index1.html')

@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/predict2',methods=['GET','POST'])
def predict2():
    print("Entered")
    
    print("Entered here")
    data = request.form['0']
    print(type(data))

    file = request.files['file'] # fet input
    filename = file.filename        
    print("@@ Input posted = ", filename)
        
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    if data == "2": 


        print("@@ Predicting class......")
        image = load_img(file_path,target_size=(128,128))
        image = img_to_array(image)
        image = image/255
        image = np.expand_dims(image,axis=0)
        
        result = np.argmax(model1.predict(image))
        print(result)
        if result == 0:
            pred = "Fake Logo"      
        elif result == 1:
            pred = "Original Logo"
        #pred, output_page = model_predict2(file_path,CTS)
                
        return render_template("result.html" , pred_output = pred, img_src=UPLOAD_FOLDER + file.filename)

    elif data == "0": 

        src = np.array(Image.open(file_path))
        src_h, src_w, _ = src.shape
        dst = seam_carving.resize(src, (128, 128),energy_mode='backward',order='width-first',keep_mask=None)
        img = Image.fromarray(dst, 'RGB')
        im1 = img.save("static/seamcarving.jpg") 
        image = img_to_array(img)
        image = image/255
        image = np.expand_dims(image,axis=0)
        
        result = np.argmax(model1.predict(image))
        print(result)
        if result == 0:
            pred = "Fake Logo"      
        elif result == 1:
            pred = "Original Logo"
        #pred, output_page = model_predict2(file_path,CTS)
                
        return render_template("result1.html" , pred_output = pred, img_src=UPLOAD_FOLDER + file.filename)

    elif data == "1": 

        img = Image.open(file_path)
        img = img.convert("L")
        # Calculating Edges using the passed laplacian Kernel
        final = img.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
                                          -1, -1, -1, -1), 1, 0))
        final.save("static/edge.png")
        image = load_img(file_path,target_size=(128,128))
        image = img_to_array(image)
        image = image/255
        image = np.expand_dims(image,axis=0)
        
        result = np.argmax(model1.predict(image))
        print(result)
        if result == 0:
            pred = "Fake Logo"      
        elif result == 1:
            pred = "Original Logo"
        #pred, output_page = model_predict2(file_path,CTS)
                
        return render_template("result2.html" , pred_output = pred, img_src=UPLOAD_FOLDER + file.filename)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    The function takes in an image, runs it through the model, and then saves the output image to a
    static folder
    :return: The image is being returned.
    """
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model(img, size=415)
        results.render()  
        for img in results.render():
            img_base64 = Image.fromarray(img)
            img_base64.save("static/image0.jpg", format="JPEG")
        return redirect("static/image0.jpg")
    return render_template("index1.html")  
        
@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(1000,5000)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "evotingotp4@gmail.com"
    msg['To'] = email
    
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("evotingotp4@gmail.com", "xowpojqyiygprhgr")
    s.send_message(msg)
    s.quit()
    return render_template("val.html")

@app.route('/predict1', methods=['POST'])
def predict1():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signup.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signin.html")

@app.route("/notebook")
def notebook1():
    return render_template("Notebook.html")



   
if __name__ == '__main__':
    app.run(debug=False)