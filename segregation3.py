import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from datetime import datetime
import easygui
import face_recognition
import os
import sys
import matplotlib.pyplot as plt

# Create an empty DataFrame to store emotion data
df = pd.DataFrame(columns=['time', 'emotion', 'person'])

# Dictionary mapping emotion labels to human-readable strings
emotion_dict = {0: 'anger', 1: 'contempt', 2: 'disgust',
                3: 'fear', 4: 'happiness',
                5: 'sadness', 6: 'surprise'}

model = load_model('model.h5')
# Dictionary to store known persons' face encodings
known_persons = {}

# Dictionary to store each person's emotions
person_emotions = {}

# Output directory for storing face images
output_directory = 'analysed_images'
os.makedirs(output_directory, exist_ok=True)

# Function to convert face image to emotion label
def convert_image(image):
    image_arr = []
    pic = cv2.resize(image, (48, 48))
    image_arr.append(pic)
    image_arr = np.array(image_arr)
    image_arr = image_arr.astype('float32')
    image_arr /= 255
    predictions = model.predict(image_arr)
    prediction = np.argmax(predictions[0])
    return emotion_dict[prediction]

# Parse user choice from command line arguments
choice = int(sys.argv[1])
#choice=int(input())
# Handle different input sources
if choice == 1:
    # Camera feed
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("================================================================================")
    print("The camera feed analysis is chosen =>")
    print("The system camera is being opened")
    print("================================================================================")
elif choice == 2:
    # Video file
    path = easygui.fileopenbox(default='*')
    cap = cv2.VideoCapture(path)
    print("================================================================================")
    print("The video at the path ", path, " is chosen")
    print("================================================================================")
elif choice == 3:
    # Image file
    j = 1
    image_path = easygui.fileopenbox(default='*')
    gray = cv2.imread(image_path)
    time_rec = datetime.now()
    print("================================================================================")
    print("The image at the path ", image_path, " is chosen")
    print("================================================================================")
    face_locations = face_recognition.face_locations(gray)

    for face_location in face_locations:
        top, right, bottom, left = face_location
        roi_gray = gray[top:bottom, left:right]
        face_encoding = face_recognition.face_encodings(gray, [face_location])[0]

        found_match = False
        for person, encodings in known_persons.items():
            if any(face_recognition.compare_faces(encodings, face_encoding, tolerance=0.6)):
                found_match = True
                break

        if not found_match:
            print("/////////////////////////////new person is detected/////////////////////////////")
            person = len(known_persons) + 1
            known_persons[person] = [face_encoding]

        emotion = convert_image(roi_gray)
        #df = df._append({'time': time_rec, 'emotion': emotion, 'person': person}, ignore_index=True)
        person_emotions[person] = person_emotions.get(person, []) + [emotion]

        person_folder = os.path.join(output_directory, f'person_{person}')
        os.makedirs(person_folder, exist_ok=True)
        cv2.imwrite(os.path.join(person_folder, f'frame_head_{person}_{len(person_emotions[person])}.jpg'), roi_gray)

        cv2.rectangle(gray, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(gray, emotion, (left + 20, top - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2, cv2.LINE_AA)
        # Generate visualizations for each detected face
        plt.figure(figsize=(10, 5))

        # Load the detected face image
        face_image = gray[top:bottom, left:right]

        # Plot the face image
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Person {person} - Face')

        # Pie Chart
        plt.subplot(1, 3, 2)
        labels, counts = zip(
            *[(emotion, person_emotions[person].count(emotion)) for emotion in set(person_emotions[person])])
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title(f'Person {person} - Dominant Emotion Pie Chart')

        # Histogram
        plt.subplot(1, 3, 3)
        sorted_emotions = sorted(set(person_emotions[person]))
        label_mapping = {emotion: i for i, emotion in enumerate(sorted_emotions)}
        numeric_labels = [label_mapping[emotion] for emotion in person_emotions[person]]
        hist, bins = np.histogram(numeric_labels, bins=len(sorted_emotions))
        plt.bar(range(len(sorted_emotions)), hist, tick_label=sorted_emotions, edgecolor='black')
        plt.title(f'Person {person} - Dominant Emotion Histogram')

        plt.tight_layout()
        plt.show()

    # print(df.head())
    # print(df.shape)

# Continue for camera feed or video file
if choice <= 2:
    frame_counter = 0
    while cap.isOpened():
        time_rec = datetime.now()
        ret, frame = cap.read()
        if ret:
            frame_counter += 1
            FPS = int(cap.get(cv2.CAP_PROP_FPS))
            gray = cv2.flip(frame, 1)
            if choice == 1 or (choice == 2 and frame_counter % FPS == 0):
                face_locations = face_recognition.face_locations(gray)

                for face_location in face_locations:
                    top, right, bottom, left = face_location
                    roi_gray = gray[top:bottom, left:right]
                    face_encoding = face_recognition.face_encodings(gray, [face_location])[0]

                    found_match = False
                    for person, encodings in known_persons.items():
                        if any(face_recognition.compare_faces(encodings, face_encoding, tolerance=0.6)):
                            found_match = True
                            break

                    if not found_match:
                        print("/////////////////////////////new person is detected/////////////////////////////")
                        person = len(known_persons) + 1
                        known_persons[person] = [face_encoding]

                    emotion = convert_image(roi_gray)
                    df = df._append({'time': time_rec, 'emotion': emotion, 'person': person}, ignore_index=True)
                    person_emotions[person] = person_emotions.get(person, []) + [emotion]

                    person_folder = os.path.join(output_directory, f'person_{person}')
                    os.makedirs(person_folder, exist_ok=True)
                    cv2.imwrite(os.path.join(person_folder, f'frame_head_{person}_{len(person_emotions[person])}.jpg'), roi_gray)
                    cv2.rectangle(gray, (left, top), (right, bottom), (255, 0, 0), 2)

            cv2.namedWindow('Video', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('Video', gray)
            cv2.resizeWindow('Video', 1000, 600)

            if cv2.waitKey(1) == 27:  # press ESC to break
                cap.release()
                cv2.destroyAllWindows()
                break

        else:
            break

    print(df.head())
    print(df.shape)

        # Visualize each person's dominant emotion
    for person, emotions in person_emotions.items():
        plt.figure(figsize=(10, 5))

        # Load the person's face image from the folder
        person_folder = os.path.join(output_directory, f'person_{person}')
        folder_files = os.listdir(person_folder)
        face_image_path = os.path.join(person_folder, folder_files[0])
        face_image = cv2.imread(face_image_path)
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Plot the face image
        plt.subplot(1, 3, 1)
        plt.imshow(face_image)
        plt.title(f'Person {person} - Face')

        # Pie Chart
        plt.subplot(1, 3, 2)
        labels, counts = zip(*[(emotion, emotions.count(emotion)) for emotion in set(emotions)])
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title(f'Person {person} - Dominant Emotion Pie Chart')

        # Histogram
        plt.subplot(1, 3, 3)
        sorted_emotions = sorted(set(emotions))
        label_mapping = {emotion: i for i, emotion in enumerate(sorted_emotions)}
        numeric_labels = [label_mapping[emotion] for emotion in emotions]
        hist, bins = np.histogram(numeric_labels, bins=len(sorted_emotions))
        plt.bar(range(len(sorted_emotions)), hist, tick_label=sorted_emotions, edgecolor='black')
        plt.title(f'Person {person} - Dominant Emotion Histogram')

        plt.tight_layout()
        plt.show()

    # Provide feedback based on detected emotions
    for person, emotions in person_emotions.items():
        dominant_emotion = max(set(emotions), key=emotions.count)

        if dominant_emotion == 'happiness':
            print(f"Person {person} seems happy!")
            # Perform actions corresponding to happiness, e.g., display a positive message

        elif dominant_emotion == 'anger':
            print(f"Person {person} seems angry!")
            # Perform actions corresponding to anger, e.g., offer assistance or calming techniques

        elif dominant_emotion == 'sadness':
            print(f"Person {person} seems sad.")
            # Perform actions corresponding to sadness, e.g., offer support or empathy

        elif dominant_emotion == 'surprise':
            print(f"Person {person} seems surprised!")
            # Perform actions corresponding to surprise, e.g., ask for clarification

        elif dominant_emotion == 'fear':
            print(f"Person {person} seems afraid.")
            # Perform actions corresponding to fear, e.g., provide reassurance

        elif dominant_emotion == 'contempt':
            print(f"Person {person} seems to be showing contempt.")
            # Perform actions corresponding to contempt, e.g., inquire about concerns

        elif dominant_emotion == 'disgust':
            print(f"Person {person} seems disgusted.")
            # Perform actions corresponding to disgust, e.g., address any issues causing discomfort

        # Provide a general response if the emotion is not one of the above
        else:
            print(f"Person {person} is expressing {dominant_emotion}.")

    # Remove the output directory and its contents
    import shutil
    #shutil.rmtree(output_directory)
