from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
from flask.wrappers import Response
from werkzeug.utils import secure_filename
import os
import cv2
import image_ops
import face_filters

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'svg', 'webp'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['SECRET_KEY'] = 'super secret key'

camera = cv2.VideoCapture(0)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def processImage(filename, operation, *args):
    print(f"Processing {filename} with {operation}")
    print(args)
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    newFilename = os.path.join(app.config['RESULT_FOLDER'], filename)
    match operation:
        case "grayc":
            resultImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        case "brightness":
            level = round(eval(args[0]) * 255/100)
            resultImg = image_ops.brightness(img, level)
            # resultImg = image_ops.brightness(img, 128)
        case "resize":
            scale = eval(args[0])/100
            resultImg = image_ops.resize_img(img, scale)
            # resultImg = image_ops.resize_img(img, 0.1)
        case "rotate":
            resultImg = image_ops.rotate(img, eval(args[0]))
        case "contenhance":
            resultImg = image_ops.contrast_enhancement(img)
        case "text":
            resultImg = image_ops.text_enhancement(img, eval(args[0]), eval(args[1]))
            # resultImg = image_ops.text_enhancement(img, 2, 2)
        case "smooth":
            resultImg = image_ops.smooth_image(img, eval(args[0]))
            # resultImg = image_ops.smooth_image(img, 7)
        case "edge":
            resultImg = image_ops.edge_detection(img)
        case "face":
            resultImg = image_ops.face_detection(img)
        case "addsap":
            resultImg = image_ops.salt_and_pepper_noise(img, eval(args[0]), eval(args[1]))
            # resultImg = image_ops.salt_and_pepper_noise(img, 0, 0.7)
        case "remsap":
            resultImg = image_ops.remove_salt_and_pepper_noise(img, eval(args[0]))
            # resultImg = image_ops.remove_salt_and_pepper_noise(img, 5)
    
    cv2.imwrite(newFilename, resultImg)
    return newFilename

def gen_frames(val):  # generate frame by frame from camera
        while True:
            # Capture frame-by-frame
            success, frame = camera.read()
            if not success:
                break
            else:
                if val!=0:
                    frame = face_filters.face_filter(frame, val)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/filters')
def filters():
    return render_template("webcam.html")


@app.route('/edit', methods=['GET', 'POST'])
def edit():
    if request.method == 'POST':
        operation = request.form['operation']
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return "error occured"

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return "Error file not selected"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            uploadFilename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(uploadFilename)
            if operation == "rotate":
                angle = request.form['rotate_angle']
                newFilename = processImage(filename, operation, angle)
            elif operation == "brightness":
                brightness = request.form['brightness_level']
                newFilename = processImage(filename, operation, brightness)
            elif operation == "resize":
                scale = request.form['resize_scale']
                newFilename = processImage(filename, operation, scale)
            elif operation == "text":
                kernel_size = request.form['text_size']
                iterations = request.form['text_iteration']
                newFilename = processImage(filename, operation, kernel_size, iterations)
            elif operation == "smooth":
                kernel_size = request.form['gaussian_size']
                newFilename = processImage(filename, operation, kernel_size)
            elif operation == "addsap":
                mean = request.form['sap_mean']
                var = request.form['sap_var']
                newFilename = processImage(filename, operation, mean, var)
            elif operation == "remsap":
                kernel_size = request.form['sap_size']
                newFilename = processImage(filename, operation, kernel_size)
            else:
                newFilename = processImage(filename, operation)

            return render_template("index.html", result_image=newFilename, image=uploadFilename)
        


@app.route('/video_feed', methods = ['POST', 'GET'])
def video_feed():
    val = 0
    if request.method == 'POST':
        if request.form['img'] == 'Dog':
            val = 1
        if request.form['img'] == 'Neon':
            val = 2
        if request.form['img'] == 'Witch':
            val = 3
    return Response(gen_frames(val),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, port=5001)