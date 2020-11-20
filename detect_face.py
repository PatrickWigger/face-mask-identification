
import cv2
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np


def classify_face(image, bndbox, model):
    # Crop the given image
    xmin = bndbox[0]
    ymin = bndbox[1]
    xmax = bndbox[2]
    ymax = bndbox[3]

    face_image = image[ymin:ymax, xmin:xmax]

    # Resize the cropped face image to the correct dimension
    face_image = cv2.resize(src=face_image, dsize=(224, 224))
    array = img_to_array(face_image)
    array = np.expand_dims(array, axis=0)

    # Make predictions
    predictions = model.predict(preprocess_input(array))

    # Find the max to determine which class was selected
    predicted_class = np.argmax(predictions)

    # Choose color and text based on predicted classification
    if predicted_class == 0:
        color = (0, 255, 0)
        text = "Mask"
    elif predicted_class == 1:
        color = (0, 0, 255)
        text = "No Mask"
    else:
        color = (0, 255, 255)
        text = "Incorrect"

    # Draw the bounding box on the image in corresponding color
    image = cv2.rectangle(img=image,
                          pt1=(xmin, ymin),
                          pt2=(xmax, ymax),
                          color=color,
                          thickness=2)

    # Draw the text on the image in corresponding color
    image = cv2.putText(img=image,
                        text=text,
                        org=(xmin, ymax + 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=color,
                        thickness=1)

    # Return image as well as predicted label
    return image, predicted_class


