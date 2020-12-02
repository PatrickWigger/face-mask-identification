import cv2
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np

# AVERAGE_X = 0
# AVERAGE_Y = 0

# Average bounding box for noise reduction
average_bbox = [0, 0, 0, 0]


def find_face(image, face_cascade, show=False):

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


# def detect_face(image, face_cascade, eye_cascade, show=False):
#
#     global AVERAGE_X, AVERAGE_Y, last_bbox
#
#     # Convert image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Convert image to black and white
#     (threshold, bw_image) = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY)
#
#     # Detect faces in the gray image
#     faces = face_cascade.detectMultiScale3(gray_image,
#                                            scaleFactor=1.05,
#                                            minNeighbors=5,
#                                            minSize=(30, 30),
#                                            flags=cv2.CASCADE_SCALE_IMAGE,
#                                            outputRejectLevels=True)
#
#     # Detect faces in the black and white image
#     faces_bw = face_cascade.detectMultiScale(bw_image, 1.05, 5)
#
#     # Initialize arrays for faces and bounding boxes
#     bbox_list = []
#
#     if len(faces) == 0 and len(faces_bw) == 0:
#         print("NO FACE FOUND")
#         return last_bbox
#
#     thresh = 40
#
#     print(faces[2])
#
#     # Draw bounding boxes on faces
#     if len(faces) != 0:
#         for (x, y, w, h) in faces[0]:
#
#             found = False
#             if len(bbox_list) != 0:
#                 for box in bbox_list:
#                     if abs(box[0] - x) < thresh or abs(box[1] - y) < thresh:
#                         found = True
#
#             if not found:
#                 bbox = [x, y, x + w, y + h]
#                 bbox_list.append(bbox)
#
#                 print("added from gray")
#
#     if len(faces_bw) != 0:
#         for (x, y, w, h) in faces_bw:
#             found = False
#             if len(bbox_list) != 0:
#                 for box in bbox_list:
#                     if abs(box[0] - x) < thresh or abs(box[1] - y) < thresh:
#                         found = True
#
#             if not found:
#                 bbox = [x, y, x + w, y + h]
#                 bbox_list.append(bbox)
#
#                 print("added from bw")
#
#
#     if show:
#         for item in bbox_list:
#             cv2.rectangle(image, (item[0] + 2, item[1] + 2), (item[2], item[3]), (255, 0, 255), 2)
#
#     noise_threshold = 50
#     indices = []
#     final = []
#
#     if AVERAGE_X == 0 and len(bbox_list) == 1:
#         AVERAGE_X = bbox_list[0][0]
#         AVERAGE_Y = bbox_list[0][1]
#     else:
#         # Attempt to decrease noise from predicted squares
#         if len(bbox_list) != 0:
#             for i in range(len(bbox_list)):
#                 if abs(bbox_list[i][0] - AVERAGE_X) > noise_threshold or abs(bbox_list[i][1] - AVERAGE_Y) > noise_threshold:
#                     indices.append(i)
#
#     for i in range(len(bbox_list)):
#         if i not in indices:
#             final.append(bbox_list[i])
#             AVERAGE_X = (AVERAGE_X + bbox_list[i][0]) / 2
#             AVERAGE_Y = (AVERAGE_Y + bbox_list[i][1]) / 2
#
#     print("avg x: " + str(AVERAGE_X) + " avg y: " + str(AVERAGE_Y))
#     print("BBOX LIST:")
#     print(bbox_list)
#     print("FINAL:")
#     print(final)
#
#     if len(final) == 0:
#         return last_bbox
#
#     last_bbox == bbox_list
#
#     return final

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


# def classify_face(image, bndbox, model):
#
#     # Loop through all faces
#     for box in bndbox:
#
#         # Crop the given image
#         xmin = box[0]
#         ymin = box[1]
#         xmax = box[2]
#         ymax = box[3]
#
#         face_image = image[ymin:ymax, xmin:xmax]
#
#         # Resize the cropped face image to the correct dimension
#         face_image = cv2.resize(src=face_image, dsize=(224, 224))
#         array = img_to_array(face_image)
#         array = np.expand_dims(array, axis=0)
#
#         # Make predictions
#         predictions = model.predict(preprocess_input(array))
#
#         # Find the max to determine which class was selected
#         predicted_class = np.argmax(predictions)
#         prob = np.amax(predictions)
#
#         # Base text
#         base_text_class = "Wearing Mask: "
#         base_text_prob = "Certainty: "
#
#         # Choose color and text based on predicted classification
#         if predicted_class == 0:
#             color = (0, 255, 0)
#             text = "Yes"
#         elif predicted_class == 1:
#             color = (0, 0, 255)
#             text = "No"
#         else:
#             color = (0, 255, 255)
#             text = "Incorrectly"
#
#         # Draw the bounding box on the image in corresponding color
#         image = cv2.rectangle(img=image,
#                               pt1=(xmin, ymin),
#                               pt2=(xmax, ymax),
#                               color=color,
#                               thickness=5)
#
#         # Draw base text
#         image = cv2.putText(img=image,
#                             text=base_text_class,
#                             org=(30, 80),
#                             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                             fontScale=1.5,
#                             color=(0, 0, 0),
#                             thickness=3)
#
#         # Draw base text
#         image = cv2.putText(img=image,
#                             text=base_text_prob,
#                             org=(30, 150),
#                             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                             fontScale=1.5,
#                             color=(0, 0, 0),
#                             thickness=3)
#
#         # Draw the text on the image in corresponding color
#         image = cv2.putText(img=image,
#                             text=text,
#                             org=(380, 80),
#                             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                             fontScale=1.5,
#                             color=color,
#                             thickness=3)
#
#         # Draw the text on the image in corresponding color
#         image = cv2.putText(img=image,
#                             text=str(prob),
#                             org=(290, 150),
#                             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                             fontScale=1.5,
#                             color=color,
#                             thickness=3)
#
#
#     # Return image as well as predicted label
#     return image


