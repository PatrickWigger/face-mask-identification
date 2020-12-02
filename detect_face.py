import cv2
import numpy as np

# Image processing
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

# Average bounding box for noise reduction
average_bbox = [0, 0, 0, 0]


def find_face(image, face_cascade):

    # These are used to greatly reduce noise in the face detection
    global average_bbox
    threshold = 100

    # Convert the given image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Attempt to detect faces in the gray image
    faces = face_cascade.detectMultiScale3(gray_image,
                                           scaleFactor=1.05,
                                           minNeighbors=7,
                                           minSize=(threshold, threshold),
                                           flags=cv2.CASCADE_SCALE_IMAGE,
                                           outputRejectLevels=True)

    # If none were found, try with black and white image
    if len(faces[0]) == 0:
        # Convert image to black and white
        (thresh, bw_image) = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY)

        # Attempt to detect faces in the black and white image
        faces = face_cascade.detectMultiScale3(bw_image,
                                               scaleFactor=1.05,
                                               minNeighbors=7,
                                               minSize=(threshold, threshold),
                                               flags=cv2.CASCADE_SCALE_IMAGE,
                                               outputRejectLevels=True)

    # See if any faces were found in either of the face detections
    if len(faces[0]) != 0:
        # Determine which face has the highest confidence value (most likely the face, not something else)
        index = np.argmax(faces[2])

        # Obtain bounding box measurements
        (x, y, w, h) = faces[0][index]
        bbox = [x, y, x + w, y + h]

        # Update the average bounding box if it is still zeros
        if average_bbox == [0, 0, 0, 0]:
            average_bbox = bbox
        else:
            # Check that the new bounding box is close enough to the average (greatly reduces noise)
            if abs(average_bbox[0] - x) < threshold and abs(average_bbox[1] - y) < threshold:
                # If it falls in this range, update average bounding box values
                average_bbox[0] = int((average_bbox[0] + x) / 2)
                average_bbox[1] = int((average_bbox[1] + y) / 2)
                average_bbox[2] = int((average_bbox[2] + x + w) / 2)
                average_bbox[3] = int((average_bbox[3] + y + h) / 2)

                # Return the found bounding box
                return bbox

    # If the detected face was too noisy (most likely not the face), return the average box up until now
    return average_bbox


def classify_face(image, bndbox, model):

    # Crop the given image
    xmin = bndbox[0]
    ymin = bndbox[1]
    xmax = bndbox[2]
    ymax = bndbox[3]

    # Crop the face out of the original image
    face_image = image[ymin:ymax, xmin:xmax]

    # Resize the cropped face image to the correct dimension
    face_image = cv2.resize(src=face_image, dsize=(224, 224))

    # Convert the given image to an array and expand dimensions
    array = img_to_array(face_image)
    array = np.expand_dims(array, axis=0)

    # Predict the class of the given face
    predictions = model.predict(preprocess_input(array))

    # Find the max to determine which class was selected
    predicted_class = np.argmax(predictions)
    prob = np.amax(predictions)

    # Base text
    base_text_class = "Wearing Mask: "
    base_text_prob = "Certainty: "

    # Choose color and text based on predicted classification
    if predicted_class == 0:
        color = (0, 255, 0)
        text = "Yes"
    elif predicted_class == 1:
        color = (0, 0, 255)
        text = "No"
    else:
        color = (0, 255, 255)
        text = "Incorrectly"

    # Draw the bounding box on the image in corresponding color
    image = cv2.rectangle(img=image,
                          pt1=(xmin, ymin),
                          pt2=(xmax, ymax),
                          color=color,
                          thickness=5)

    # Draw base text for mask status
    image = cv2.putText(img=image,
                        text=base_text_class,
                        org=(30, 80),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.5,
                        color=(0, 0, 0),
                        thickness=3)

    # Draw base text for certainty in result
    image = cv2.putText(img=image,
                        text=base_text_prob,
                        org=(30, 150),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.5,
                        color=(0, 0, 0),
                        thickness=3)

    # Draw the mask class on the image
    image = cv2.putText(img=image,
                        text=text,
                        org=(380, 80),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.5,
                        color=color,
                        thickness=3)

    # Draw the prediction certainty on the image
    image = cv2.putText(img=image,
                        text=str(prob),
                        org=(290, 150),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.5,
                        color=color,
                        thickness=3)

    # Return image
    return image