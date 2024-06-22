import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import base64
import numpy as np
import face_recognition
from yolo_face_recognition import load_yolo_model, detect_faces_yolo

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/images'
app.config['TEMP_FOLDER'] = './static/temp'

# Vérification et création des dossiers de téléchargement et temporaire
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

students = []

# Charger le modèle YOLO
net, output_layers = load_yolo_model()

def draw_boxes(image, boxes):
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('Detected Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'POST':
        name = request.form['name']
        student_id = request.form['student_id']
        photo = request.files['photo']
        if photo:
            photo_path = os.path.join(app.config['UPLOAD_FOLDER'], photo.filename)
            photo.save(photo_path)
            students.append({
                'name': name,
                'student_id': student_id,
                'photo_path': photo_path,
                'encoding': face_recognition.face_encodings(face_recognition.load_image_file(photo_path))[0]
            })
            return redirect(url_for('index'))
    return render_template('add_student.html')

@app.route('/take_attendance')
def take_attendance():
    return render_template('take_attendance.html')

@app.route('/process_attendance', methods=['POST'])
def process_attendance():
    data = request.get_json()
    if 'image' in data:
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        indexes, boxes, confidences = detect_faces_yolo(net, output_layers, img_np)

        # Afficher les boîtes de détection sur l'image capturée
        draw_boxes(img_np, boxes)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                crop_img = img_np[y:y + h, x:x + w]
                temp_img_path = os.path.join(app.config['TEMP_FOLDER'], 'attendance_face.jpg')
                cv2.imwrite(temp_img_path, crop_img)

                student_name = recognize_student(temp_img_path)

                if student_name:
                    return jsonify({'name': student_name})
                else:
                    return jsonify({'name': 'Inconnu'})

    return jsonify({'name': 'Inconnu'})

def recognize_student(image_path):
    known_students = {student['name']: student['encoding'] for student in students}

    unknown_image = face_recognition.load_image_file(image_path)
    unknown_encodings = face_recognition.face_encodings(unknown_image)

    if not unknown_encodings:
        return 'Inconnu'

    unknown_encoding = unknown_encodings[0]

    for name, known_encoding in known_students.items():
        results = face_recognition.compare_faces([known_encoding], unknown_encoding)
        if results[0]:
            return name

    return 'Inconnu'

if __name__ == '__main__':
    app.run(debug=True)
