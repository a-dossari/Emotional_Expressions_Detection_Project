# Import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model


# Define CascadeClassifier
face_classifier = cv2.CascadeClassifier(r'C:\Users\ALDOSSARI\Desktop\Emotion_Detection/haarcascade_frontalface_default.xml')

# Define the best model
classifier =load_model(r'C:\Users\ALDOSSARI\Desktop\Emotion_Detection/model.h5')

# Define the emotional expressions
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# Define a video as input
cap = cv2.VideoCapture('Job_Interview.mp4')


# Define counters for emotional expressions
count_Angry = 0
count_Disgust = 0
count_Fear = 0
count_Happy = 0
count_Neutral = 0
count_Sad = 0
count_Surprise = 0
count_All = 0


while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)

            # count each emotional expression that appeared in the video 
            if label == 'Angry':
                count_Angry +=1
            
            elif label == 'Disgust':
                count_Disgust +=1
                
            elif label == 'Fear':
                count_Fear +=1
                
            elif label == 'Happy':
                count_Happy +=1
                
            elif label == 'Neutral':
                count_Neutral +=1
                
            elif label == 'Sad':
                count_Sad +=1
                
            elif label == 'Surprise':
                count_Surprise +=1
            
            
            
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# Number of all emotional expressions in the video
count_All = count_Angry + count_Disgust + count_Fear + count_Happy + count_Neutral + count_Sad + count_Surprise

# Percentage of each emotional expression in the video
perc_Angry = count_Angry/count_All
perc_Disgust = count_Disgust/count_All
perc_Fear = count_Fear/count_All
perc_Happy = count_Happy/count_All
perc_Neutral = count_Neutral/count_All
perc_Sad = count_Sad/count_All
perc_Surprise = count_Surprise/count_All


perc_values = [perc_Angry , perc_Disgust , perc_Fear , perc_Happy , perc_Neutral , perc_Sad , perc_Surprise]
perc_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
colors = sns.color_palette('bright')[0:5]

# Plot percentages of the emotional expressions
pie_chart = plt.pie(perc_values, labels = perc_labels, colors = colors)
plt.title("Percentages of emotions")
plt.show()


counts_values = [count_Angry , count_Disgust , count_Fear , count_Happy , count_Neutral , count_Sad , count_Surprise]

# Plot counters of the emotional expressions
bar_chart = plt.bar(x = emotion_labels, height = counts_values)
plt.xlabel("Emotions")
plt.ylabel("Number of times emotions appeared")
plt.title("Number of times each emotion has appeared")
plt.show()