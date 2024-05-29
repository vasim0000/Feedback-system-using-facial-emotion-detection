# Feedback-system-using-facial-emotion-detection
A Model which can analyze emotions of persons in images, videos and live video. This model can analyze multiple faces present in the source. Results are predicted using - Sequential Model for image analysis and emotion detection. The modules used in this project include OpenCV, Face Encodings, Face Recognition.
You can download the CK+ 48 dataset used iin this project from here https://www.kaggle.com/datasets/gauravsharma99/ck48-5-emotions
The model.h5 is trained sequential model and code for training is model_training.ipynb
The dataset we use contains 5 types of emotions anger, fear, happy, sadness and surprise.
The main advantage of this model is that, each persons emotion in a video are analysed seperately meaning that the feed back report is give to each individual in the frame.
The final feedback is visualized at last.

1. The User interface is as follows (we should run the UI.py file) :
   ![Screenshot (129)](https://github.com/vasim0000/Feedback-system-using-facial-emotion-detection/assets/84614077/0f965e76-922d-4b69-a953-9f1e36224652)

2. After selecting the input option, the segregation.py file starts its execution.
3. A new window created using the tkinter shows the analysis of the input, when ever a new person is detedted then it shows a message on the window.
   ![Screenshot (140)](https://github.com/vasim0000/Feedback-system-using-facial-emotion-detection/assets/84614077/3bd5602b-d118-49b9-8935-0ee0dbc87f6c)
4.Option 1: Choosing image as input source will provide output analysis as :
   ![Screenshot (116)](https://github.com/vasim0000/Feedback-system-using-facial-emotion-detection/assets/84614077/db69aed4-42d2-4e4d-bb7b-3c8c384ac919)
5.Option 2: Choosing Video or Live video as input source will provide output analysis on each and every face detected throughout the video until esc is clicked.The process is done by collecting frames from the video or live video and performing image analysis on it for emotion detection

For example : Below is one of the frame from a meeting video -
   ![Screenshot (108)](https://github.com/vasim0000/Feedback-system-using-facial-emotion-detection/assets/84614077/595ebd30-7a71-405b-9e78-4b30b502dd1a)
After clicking the esc button the analysis stops and all the emotions of each person present in source throughout the playing time of video are visualized as below-

![Screenshot (147)](https://github.com/vasim0000/Feedback-system-using-facial-emotion-detection/assets/84614077/2901e219-0dc8-4685-a4d1-e20bad1c9bbf)
![Screenshot (148)](https://github.com/vasim0000/Feedback-system-using-facial-emotion-detection/assets/84614077/cb1cb0ef-3be4-4d5b-9374-6868437a364a)
![Screenshot (149)](https://github.com/vasim0000/Feedback-system-using-facial-emotion-detection/assets/84614077/c61eb64a-0a5c-4d58-8545-977e80eafdac)
![Screenshot (150)](https://github.com/vasim0000/Feedback-system-using-facial-emotion-detection/assets/84614077/a1bbc926-7cc3-4637-945a-8b1c2f53a3d0)
![Screenshot (151)](https://github.com/vasim0000/Feedback-system-using-facial-emotion-detection/assets/84614077/a6e521a5-7545-46ba-a368-130de5c36a5b)
![Screenshot (152)](https://github.com/vasim0000/Feedback-system-using-facial-emotion-detection/assets/84614077/d6cf4920-5fae-4b4e-86ab-2e959dabf360)
![Screenshot (153)](https://github.com/vasim0000/Feedback-system-using-facial-emotion-detection/assets/84614077/7e925426-f013-4f06-9b2d-4ed360c77f8e)

6.At last person wise feed back report is shown on the window :-
![Screenshot (154)](https://github.com/vasim0000/Feedback-system-using-facial-emotion-detection/assets/84614077/f96af5e6-d753-4e3c-8922-0c1b58faeb7a)








