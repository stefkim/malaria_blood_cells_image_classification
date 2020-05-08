from flask import Flask, redirect, url_for, request, render_template

from pathlib import Path
from werkzeug.utils import secure_filename


from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *

UPLOAD_FOLDER = 'static/images/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


path = Path('')
classes = ['Parasitized', 'Uninfected']
data = ImageDataBunch.single_from_classes(path,
                                          classes,
                                          ds_tfms=get_transforms(do_flip = True,
                                                                 flip_vert = True,
                                                                 max_warp=0),
                                          size=224
                                          ).normalize(imagenet_stats)
model = cnn_learner(data,models.resnet34)
model.load('stage1')


def predict(file_path):
    img = open_image(file_path)
    return model.predict(img)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def upload_data():
    if request.method == 'POST':
        image = request.files['file']
        if image.filename == '':
            return render_template('index.html',
                                   err = True
                                   )
        else:
            img_file = request.files['file']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img_file.filename))
            img_file.save(img_path)

            prediction = predict(img_path)

            # return data_path,predict(data_path)
            return render_template('index.html',
                                   result_class=str(prediction[0]),
                                   result_accuracy = '{:.3f}%'.format(float(max(prediction[2]))*100),
                                   result_image = img_path,
                                   show_modal = True)
            #return str(prediction[0])

    return None

if __name__ == '__main__':
    app.run()
