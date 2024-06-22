import os
import cv2
import numpy as np


def load_yolo_model():
    # Chemins complets vers les fichiers de configuration et de poids
    config_path = r'C:\Users\ERRIHANI  Hicham\Downloads\yolov3.cfg'
    weights_path = r'C:\Users\ERRIHANI  Hicham\Downloads\yolov3.weights'

    # Imprimer les chemins pour le débogage
    print(f"Chemin de la configuration : {config_path}")
    print(f"Chemin des poids : {weights_path}")

    # Vérifiez si les fichiers existent
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Fichier de configuration non trouvé : {config_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Fichier de poids non trouvé : {weights_path}")

    # Charger le modèle YOLO
    net = cv2.dnn.readNet(weights_path, config_path)

    # Obtenir les noms des couches de sortie
    layer_names = net.getLayerNames()

    # Débogage: imprimer les noms des couches
    print(f"Noms des couches : {layer_names}")

    unconnected_out_layers = net.getUnconnectedOutLayers()

    # Débogage: imprimer les couches de sortie non connectées
    print(f"Couches de sortie non connectées : {unconnected_out_layers}")

    # Ajustez l'indexation en fonction de la sortie
    if isinstance(unconnected_out_layers, np.ndarray):
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
    else:
        output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]

    return net, output_layers


def detect_faces_yolo(net, output_layers, image):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return indexes, boxes, confidences


# Charger le modèle YOLO et vérifier les couches de sortie
net, output_layers = load_yolo_model()
