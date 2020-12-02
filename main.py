from create_model import *
from train_model import *
from detect_face import *
from os import path
import cv2

from keras.models import load_model

# Specify the type of model to be used (VGG16, ResNet50, MobileNetV2)
MODEL = "VGG16"

# Directory prefix to hold the models after saving
model_directory_prefix = "models/"

# Path to video, if there is one
video_path = "demo2.mov"

# Path to image, if there is one
image_path = "data/images/maksssksksss13.png"


def main():

    # Specify to use video input or photo input
    use_video = True

    # Ensure the specified model is one of the valid options
    assert(MODEL in ("MobileNetV2", "ResNet50", "VGG16"))

    # Determine if the model already exists
    if not path.exists(model_directory_prefix + MODEL + "/model"):
        # Create the specified model
        print("Creating " + MODEL + " model...")

        # Read the images and parse the XML annotations to find labels and bounding boxes for faces
        image_annotations = read_images()

        # Crop the found faces with bounding boxes and label them
        faces, labels = find_faces(data=image_annotations, model=MODEL)

        # Create the specified model
        model = create_model(model_type=MODEL)

        # Train the model on the data
        model = train_model(model=model, data=faces, labels=labels, test_size=0.4, epochs=2)

        # Save the model to the specified directory
        print("Saving " + MODEL + " model...")
        model.save(model_directory_prefix + MODEL + "/model")
    else:
        # Load the model from memory
        print("Loading " + MODEL + " from memory...")
        model = load_model(model_directory_prefix + MODEL + "/model")

        # Initialize face detector using OpenCV Cascade Classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Run face and mask detection on a video input
        if use_video:

            # Initialize video capture
            video_capture = cv2.VideoCapture(video_path)
            got_image, frame = video_capture.read()

            # Check if the video is read successfully
            if not got_image:
                print("Cannot read video source!")
                exit(0)

            # Obtain height and width of video
            video_height = frame.shape[0]
            video_width = frame.shape[1]

            # Initialize video writer to save output video
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            videoWriter = cv2.VideoWriter("detect.avi", fourcc=fourcc, fps=20.0, frameSize=(video_width, video_height))

            # Loop through all video frames
            while True:
                # Obtain the next frame of the video
                got_image, frame = video_capture.read()
                if not got_image:
                    break

                # Detect faces in the video
                bbox_list = find_face(image=frame, face_cascade=face_cascade)

                # Determine the classification of the face
                frame = classify_face(image=frame, bndbox=bbox_list, model=model)

                # Write the given frame to the output video
                output = frame.copy()
                videoWriter.write(output)

            # Save the video
            videoWriter.release()

        else:
            # Load image from a specified path
            image = cv2.imread(image_path)

            # Detect face in the image
            bbox_list = find_face(image=image, face_cascade=face_cascade)

            # Determine the classification of the face
            image = classify_face(image=image, bndbox=bbox_list, model=model)

            cv2.imshow("Predicted Class", image)
            cv2.waitKey(0)


if __name__ == '__main__':
    main()