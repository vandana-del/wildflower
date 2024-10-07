from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
import mysql.connector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load the AI model
model = tf.keras.models.load_model('model/model.h5')

# Connect to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Prameela@14",
    database="wildflower_db"
)

def classify_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return np.argmax(predictions)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        flower_class = classify_image(filepath)

        cursor = db.cursor()
        cursor.execute("SELECT medicinal_use FROM flowers WHERE id = %s", (flower_class,))
        result = cursor.fetchone()

        return render_template('result.html', flower_class=flower_class, medicinal_use=result[0])

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

