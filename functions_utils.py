from tensorflow import keras
import tensorflow as tf

from matplotlib import colors
import numpy as np

import os

model_path = "/home/Jupiter/OpenClassroom-Projet-8-Participez-la-conception-d-une-voiture-autonome/model_fcn8_no_augm.h5"
#model_path = "D:/anaconda3\envs\env1/notebooks\OP Notebooks\p8\Github/model_fcn8_no_augm.h5"

assert os.path.exists(model_path), f"Le fichier de modèle '{model_path}' n'existe pas."



# Fonction pour normaliser une image d'entrée dans la plage [-1, 1]
def normalize_input_img(img):
    '''
    Args:
    img (PIL.Image): L'image PIL à normaliser
    
    Returns:
    numpy.ndarray: Image normalisée sous forme d'un tableau 3D numpy
    '''
    # Convertir l'image PIL en tableau numpy
    img = tf.keras.preprocessing.image.img_to_array(img, dtype=np.int32)
    
    # Normaliser l'intensité des pixels dans la plage [-1, 1]
    img = img / 127.5
    img -= 1
    
    return img

cats = {
    'void': [0, 1, 2, 3, 4, 5, 6],
    'flat': [7, 8, 9, 10],
    'construction': [11, 12, 13, 14, 15, 16],
    'object': [17, 18, 19, 20],
    'nature': [21, 22],
    'sky': [23],
    'human': [24, 25],
    'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]
}

# Fonction pour effectuer les prédictions
def predict(image_path):


    model = keras.models.load_model(model_path, compile = False)
    model.compile(loss='categorical_crossentropy',
            metrics=['accuracy'])

    
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128,256))
    # Normaliser l'image d'entrée
    img_norm = np.array(normalize_input_img(img))  
    
    img_norm = img_norm.reshape(1, 128, 256, 3)
    

    # Prédire le masque segmenté à l'aide du modèle
    img_pred = model.predict(img_norm)[0]

    return generate_img_from_mask(img_pred, cats)



    

# Fonction pour générer une image à partir d'un masque segmenté
def generate_img_from_mask(mask, cats):
    '''
    Args:
    mask (numpy.ndarray): Masque segmenté sous forme d'un tableau numpy
    cats (dict): Dictionnaire de catégories avec des palettes de couleurs associées
    
    Returns:
    PIL.Image: Image générée à partir du masque segmenté
    '''

    colors_palette = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    # Créer une image vide avec les dimensions du masque
    img_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype='float')

    # Remplir l'image générée avec les couleurs de catégorie du masque
    for cat in range(len(cats)):
        img_seg[:, :, 0] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[0]
        img_seg[:, :, 1] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[1]
        img_seg[:, :, 2] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[2]

    # Convertir le tableau numpy en image PIL
    return tf.keras.preprocessing.image.array_to_img(img_seg)