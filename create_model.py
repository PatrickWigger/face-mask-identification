import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# Model Layers
from keras.models import Model
from keras.layers import Input
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


from keras.preprocessing.image import img_to_array

# Keras models
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16

# Import correct preprocess function based on model
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input

data_directory_prefix = "data/"
IMG_SIZE = 224


def find_faces(data, model, show_boxes=False):
    # Hold images of all faces and their corresponding mask labels
    faces = []
    labels = []

    assert model in ("MobileNetV2", "VGG16", "ResNet50")

    # Iterate through all image information
    for image in data:
        # Read the image
        bgr_image = cv2.imread(filename=data_directory_prefix + "images/" + image["name"])

        # Locate each face using the bounding boxes
        for bndbox, label in zip(image["bndboxes"], image["classes"]):
            xmin = bndbox[0]
            ymin = bndbox[1]
            xmax = bndbox[2]
            ymax = bndbox[3]

            # To reduce training time, only use faces larger than 40 by 40 pixels
            limit = 40
            if (xmax - xmin) > limit and (ymax - ymin) > limit:

                # Crop the face image
                face_image = bgr_image[ymin:ymax, xmin:xmax]

                # Resize the face image
                face_image = cv2.resize(src=face_image, dsize=(IMG_SIZE, IMG_SIZE))

                # Preprocess the input based on the model to be used
                face_image = img_to_array(img=face_image)

                if model == "ResNet50":
                    face_image = resnet50_preprocess_input(face_image)
                elif model == "VGG16":
                    face_image = vgg16_preprocess_input(face_image)
                elif model == "MobileNetV2":
                    face_image = mobilenet_preprocess_input(face_image)

                faces.append(face_image)
                labels.append(label)

            if show_boxes:
                # Choose a color based on the class
                if label == 0:
                    color = (0, 255, 0)
                elif label == 1:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 255)

                # Optionally draw a rectangle around the face
                bgr_image = cv2.rectangle(img=bgr_image,
                                          pt1=(xmin, ymin),
                                          pt2=(xmax, ymax),
                                          color=color,
                                          thickness=2)

        if show_boxes:
            cv2.imshow("image", bgr_image)
            cv2.waitKey(0)

    faces = np.array(faces, dtype=np.float32)
    labels = np.array(labels)

    return faces, labels


def read_images():
    # Load the paths for annotations and images
    annotation_paths = list(sorted(os.listdir(data_directory_prefix + "annotations")))
    image_paths = list(sorted(os.listdir(data_directory_prefix + "images")))

    # Will hold annotation information for all images
    data = []

    # Grab each XML annotation file and corresponding image
    for annotation, image in zip(annotation_paths, image_paths):
        # Check to make sure these refer to the same image
        assert(annotation[:-4] == image[:-4])

        # Read the annotations (XML file) using element tree
        xml_tree = ET.parse(data_directory_prefix + "annotations/" + annotation)
        xml_root = xml_tree.getroot()

        # Create output package for image
        image_data = {}

        # Will hold bounding boxes and classes for each face in the image
        all_bndboxes = []
        all_classes = []

        for category in xml_root:
            # Check for object tag, representing an individual face
            if category.tag == "object":
                # Initialize the output variables
                bndbox = [0, 0, 0, 0]
                classification = -1

                # Determine the category for the image
                name = category[0].text

                # Class 0: Mask On, Class 1: No Mask, Class 2: Incorrectly Worn Mask
                if name == "with_mask":
                    classification = 0
                elif name == "without_mask":
                    classification = 1
                elif name == "mask_weared_incorrect":
                    classification = 2
                else:
                    classification = 3

                # Check that the image is identified in one of the classes
                assert(classification in range(0, 3))

                # Obtain bounding box sizes
                for dim in category[5]:
                    # Double check that this is the bbox category
                    assert(category[5].tag == "bndbox")

                    if dim.tag == "xmin":
                        bndbox[0] = int(dim.text)
                    elif dim.tag == "ymin":
                        bndbox[1] = int(dim.text)
                    elif dim.tag == "xmax":
                        bndbox[2] = int(dim.text)
                    elif dim.tag == "ymax":
                        bndbox[3] = int(dim.text)

                # Add the found classification and bndbox to the total for the image
                all_classes.append(classification)
                all_bndboxes.append(bndbox)

        # Add the XML annotations to the dictionary
        image_data['name'] = image
        image_data['bndboxes'] = all_bndboxes
        image_data['classes'] = all_classes
        data.append(image_data)

    # Return list of all image annotations
    return data


def create_model(model_type):

    if model_type == "VGG16":
        # Use VGG16 model with ImageNets weights
        model = VGG16(weights="imagenet",
                      include_top=False,
                      input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    elif model_type == "ResNet50":
        # Use ResNet50 model with ImageNet weights
        model = ResNet50(weights="imagenet",
                         include_top=False,
                         input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    else:
        # Use MobileNetV2 model with ImageNet weights
        model = MobileNetV2(weights="imagenet",
                            include_top=False,
                            input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

    # Create shallow of layers to prepend to deep layers
    layers = model.output
    layers = AveragePooling2D(pool_size=(7, 7))(layers)
    layers = Flatten()(layers)
    layers = Dense(units=64, activation="relu")(layers)
    layers = Dropout(rate=0.4)(layers)
    layers = Dense(units=3, activation="softmax")(layers)

    # Construct a new network with these layers
    cnn = Model(inputs=model.inputs, outputs=layers)

    for l in model.layers:
        l.trainable = False

    return cnn







