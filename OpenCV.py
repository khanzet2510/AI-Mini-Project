from keras.models import load_model 
import cv2
import numpy as np
import csv
from datetime import datetime
from datetime import date
import cvzone

#Set_cam
cap = cv2.VideoCapture(0)
ret, img = cap.read()
imgFront = cv2.imread('image2.png', cv2.IMREAD_UNCHANGED)
hf, wf, cf = imgFront.shape
hb, wb, cb = img.shape
#Set_model
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

#Generate dictionary
dic_date={}

face_cap=cv2.CascadeClassifier("C:/Users/Admin/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")


#Save csv
def save_csv():
    datetime=date.today()
    header=['Name\t', 'Time \t']
    f= open(f"C:/Users/Admin/OneDrive/Python+SQL/AI Mini Project/Attendance/Attendance date {datetime}.csv","a+" )
    writer=csv.writer(f)
    writer.writerow(header)
    for key, value in dic_date.items():
        writer.writerow([key,value]) 
    dic_date.clear()
    f.close()

#Generate loop
while True:
    ret, img = cap.read()
    img=cv2.flip(img,1)
    

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Detect faces in the frame
    if int(str(np.round(confidence_score * 100))[:-2]) <95:
        name = 'Unknown'
        color=(0,0,255)
    else:
        name= f"{class_name[2:-1]} {str(np.round(confidence_score * 100))[:-2]}%"
        color=(0, 255, 0)
        current_time = datetime.now().strftime("%H:%M:%S")
        dic_date[class_name[2:-1]]=current_time
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img,name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2,2)
       
    imgResult = cvzone.overlayPNG(img, imgFront, [0, hb-hf])
    #Open cam
    cv2.imshow("Video", imgResult)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)
    print(dic_date)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        save_csv()
        break
    
cap.release()
cv2.destroyAllWindows()




