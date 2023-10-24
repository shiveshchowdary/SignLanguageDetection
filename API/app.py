from flask import Flask, render_template, request, Response, redirect, url_for
import cv2
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
import random
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

from datetime import datetime



app = Flask(__name__, static_folder='static')
model = keras.models.load_model('API\\Signmodel.h5')
# Load the pre-trained ResNet50 model
model1 = ResNet50(weights='imagenet', include_top=False, pooling='avg')
Message = " "


def compare_images():
    img_path = "API\\Images\\temp_sign.png"
    image2_path = "API\\Images\\Right_hand.jpg"
    image1_path = "API\\Images\\left_hand.jpg"
    # Load the two input images
    img1 = image.load_img(img_path, target_size=(224, 224))
    img2r = image.load_img(image2_path, target_size=(224, 224))
    img2l = image.load_img(image1_path, target_size=(224, 224))

    # Preprocess the images
    x1 = image.img_to_array(img1)
    x1 = np.expand_dims(x1, axis=0)
    x1 = preprocess_input(x1)

    x2 = image.img_to_array(img2r)
    x2 = np.expand_dims(x2, axis=0)
    x2 = preprocess_input(x2)

    x3 = image.img_to_array(img2l)
    x3 = np.expand_dims(x3, axis=0)
    x3 = preprocess_input(x3)

    # Use the ResNet50 model to extract features from the images
    features1 = model1.predict(x1)
    features2 = model1.predict(x2)
    features3 = model1.predict(x3)

    # Compute the cosine similarity between the feature vectors
    similarity1 = np.dot(features1, features2.T) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    similarity2 = np.dot(features1,features3.T)/(np.linalg.norm(features1)*np.linalg.norm(features3))
    # Return the similarity value
    if max(similarity1[0][0],similarity2[0][0]) > 0.7:
        return 1
    return 0


def fetch_img(S):
    S = S.lower()
    images = []
    for i in S:
        if i >= 'a' and i <= 'z':
            if i != 'j' and i != 'z':
                for j in range(6):
                    images.append(i+'.png')
            elif i == 'j':
                for j in range(3):
                    for z in range(2):
                        images.append(i+'000'+str(j)+'.png')
            else:
                for j in range(1,7):
                    images.append(i+'000'+str(j)+'.png')
        if i == ' ':
            for j in range(6):
                images.append("sp.png")

    return images


def generate_video(S):
    image_folder = 'API\\Images' 
    video_name = "API\\static\\ChToSign.mp4"
    images = fetch_img(S)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 6, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()


def predict(img):
    Other_letter = ['j','z']
    img.resize(1, 28, 28, 1)
    y = model.predict(img)
    if max(y[0]) < 0.96:
        y_pred = random.randint(0, 1)
        y_pred = Other_letter[y_pred]
        return y_pred
    y_pred = np.argmax(y)

    y_pred = y_pred + ord('a')
    y_pred = chr(y_pred)
    
    return y_pred

def Hand_img_alphabet(img,i):
    # Load the input image
#   img = cv2.imread(image_path)

  # Convert the image to grayscale
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Apply a median blur to reduce noise
  gray_img = cv2.medianBlur(gray_img, 5)

  # Detect skin color regions
  lower_skin = np.array([0, 20, 70], dtype=np.uint8)
  upper_skin = np.array([20, 255, 255], dtype=np.uint8)
  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv_img, lower_skin, upper_skin)

  # Find contours in the skin color mask
  contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # Find the largest contour (i.e., the hand)
  if len(contours) > 0:
      largest_contour = max(contours, key=cv2.contourArea)
      
      # Find the convex hull of the hand
      hull = cv2.convexHull(largest_contour)

      # Find the top, bottom, left, and right points of the hull
      top = tuple(hull[hull[:, :, 1].argmin()][0])
      bottom = tuple(hull[hull[:, :, 1].argmax()][0])
      left = tuple(hull[hull[:, :, 0].argmin()][0])
      right = tuple(hull[hull[:, :, 0].argmax()][0])
      
      # Find the wrist point
      wrist = ((left[0] + right[0]) // 2, bottom[1])

      # Crop the hand image
      img = img[top[1]:wrist[1], left[0]:right[0]]
  cv2.imwrite("API\\Images\\temp_sign.png", img)
  if compare_images() == 1:
        return " "
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  resized_img = cv2.resize(gray_img, (28, 28), interpolation=cv2.INTER_AREA)
#   cv2.imwrite(str(i)+'.jpg', resized_img)
  return predict(resized_img)

def gen_frames():
    Signs = []
    Message = ''
    camera = cv2.VideoCapture(0)
    i = 0
    Frame = None
    while True:
        
        i += 1
        success, frame = camera.read()
        F1 = frame
        if not (success):
            break
        else:
            # encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            if (i-1) % 120 == 0:
                Frame = frame
                x = Hand_img_alphabet(F1,i)
                Signs.append(x)
                Message += x
                print(Message)
                

                # now = datetime.now()
                # current_time = now.strftime("%H:%M:%S")
                # print("Current Time =", current_time)
            # yield the frame to be displayed in the web page
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + Frame + b'\r\n')
        
        i += 1
    camera.release()


    


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html',dataToRender=Message,show = True)

# @app.route('/msg')
# def msg():
#     data = Message
#     return render_template('index.html', dataToRender=data,show = True,msg =True)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    # get the image from the camera
    camera = cv2.VideoCapture(0)
    _, image = camera.read()
    camera.release()

    # generate a unique filename and save the image
    filename = 'image.jpg'
    cv2.imwrite("static\\"+filename, image)

    # redirect to a page that displays the captured image
    return redirect(url_for('show_image', filename=filename))

@app.route('/show_image/<filename>')
def show_image(filename):
    # display the captured image
    return render_template('index.html',dataToRender=Message, filename=filename, Ret = False, show = False, strt = True, cap = True)


S = None
@app.route('/',methods=['POST'])
def Show():
    S = request.form['files']
    generate_video(S)
    return render_template('index.html', dataToRender=Message,Ret = True, show = True, strt = False, cap = False)

@app.route('/capture_start/',methods=['POST'])
def Show1():
    # do a check if Video.mp4 file excists or not
    return render_template('index.html', dataToRender=Message,Ret = False, show = False, strt = True, cap = True)

@app.route('/capture_end/',methods=['POST'])
def Show2():
    return render_template('index.html', dataToRender=Message,Ret = False, show = True, strt = False, cap = False)





if __name__ == '__main__':
    app.run(port = 3000, debug = True)