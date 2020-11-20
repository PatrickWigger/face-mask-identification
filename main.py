from create_model import *
from train_model import *
from detect_face import *
from os import path
import cv2

from keras.models import load_model

# Specify the type of model to be used (VGG16, ResNet50, MobileNetV2)
MODEL = "VGG16"

# Directory prefix to hold trained models
model_directory_prefix = "models/"


def main():

    assert(MODEL in ("MobileNetV2", "ResNet50", "VGG16"))

    # Determine if the model exists
    if not path.exists(model_directory_prefix + MODEL + "/model"):
        # Create the model
        print("Creating " + MODEL + " model...")

        # Read the images and parse the XML annotations to find labels and bounding boxes for faces
        data = read_images()

        # Crop the found faces with bounding boxes and label them
        faces, labels = find_faces(data=data, model=MODEL)

        # Create the specified model
        model = create_model(model_type=MODEL)

        # Train the model on the data
        model = train_model(model=model, data=faces, labels=labels, test_size=0.4, epochs=2)

        print("Saving " + MODEL + " model...")

        model.save(model_directory_prefix + MODEL + "/model")
    else:
        # Load the model
        print("Loading " + MODEL + " from memory...")
        model = load_model(model_directory_prefix + MODEL + "/model")

        # TODO obtain images from video feed and detect face bounding box

        # Test image and bounding box from dataset
        test_image = cv2.imread("data/images/maksssksksss0.png")
        test_bndbox = [185, 100, 226, 144]

        # Determine the classification (0 = mask, 1 = no mask, 2 = incorrectly worn mask)
        image, classification = classify_face(image=test_image, bndbox=test_bndbox, model=model)

        cv2.imshow("Predicted Class", image)
        cv2.waitKey(0)
        print("Predicted Class: " + str(classification))


if __name__ == '__main__':
    main()