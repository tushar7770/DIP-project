from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'svg', 'webp'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['SECRET_KEY'] = 'super secret key'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def processImage(filename, operation):
    print(f"Processing {filename} with {operation}")
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    match operation:
        case "grayc":
            resultImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            newFilename = os.path.join(app.config['RESULT_FOLDER'], filename)
            cv2.imwrite(newFilename, resultImg)
            return newFilename


@app.route('/')
def home():
    return render_template("index.html")


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
            newFilename = processImage(filename, operation)
            return render_template("index.html", result_image=newFilename, image=uploadFilename)


if __name__ == '__main__':
    app.run(debug=True, port=5001)