from keras.preprocessing import image
import numpy as np
import os
import cv2
import tensorflow as tf
import pandas as pd

#folder path with images
folder_path = '/eval_data/NG'
# path to model
model_path = '/01_LEGO.hdf5'

# dimensions of images
img_width, img_height = 160, 160

# load the trained model
model = tf.keras.models.load_model(model_path)

# load all images into a list
images = []

i = 0

for img in os.listdir(folder_path):
    filename = os.path.join(folder_path, img)
    img = image.load_img(filename, target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    classes = model.predict_classes(img, batch_size=16)
    #prediction = pd.DataFrame(classes, columns=['predictions']).to_csv('prediction.csv')
    if classes==0:
        print("NG")
    elif classes==1:
        print("OK")

    #Uncomment to classify into folders
    i += 1
    # if classes == 0:
    #     filename1 = result_negative_folder + "/image_" + str(i) + ".jpg"
    #     cv2.imwrite(filename1, img)
    # elif classes == 1:
    #     filename2 = result_positive_folder + "/image_" + str(i) + ".jpg"
    #     cv2.imwrite(filename2, img)
