from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import model_runner as mr

model=None
config={}
tf=None

def init_flask(model_ext, config_ext, tf_ext):
    global model, config, tf
    app = Flask(__name__)
    model=model_ext
    config=config_ext
    if tf is None:
        tf=tf_ext

    UPLOAD_FOLDER = 'UserData'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.config['run.upload_folder'] = UPLOAD_FOLDER

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/upload', methods=['POST'])
    def upload():
        global model
        if 'photo' not in request.files:
            return "No file part", 400

        file = request.files['photo']

        if file.filename == '':
            return "No selected file", 400

        # Get the file extension
        ext = os.path.splitext(secure_filename(file.filename))[1]
        filename = f"f1{ext}"
        filepath = os.path.join(app.config['run.upload_folder'], filename)

        file.save(filepath)

        classes=config["pre.classes"].split(",")
        classification_prediction=None
        max=0
        predictions=mr.run_model(model, filepath, config, tf)[0]
        result=None
        try:
            for i in range(0,len(predictions)):
                if predictions[i]>float(config["run.threshold"]) and predictions[i]>max:
                    max=predictions[i]
                    classification_prediction=i
            
            if classification_prediction is None:
                result="Uncertain"
            else:
                result=classes[classification_prediction]
        except Exception as e:
            raise e

        return f"Model thinks `{result}` with {max*100}% confidence.", 200

    app.run(debug=False)
