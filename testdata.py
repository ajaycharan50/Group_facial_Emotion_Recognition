import cv2
import numpy as np
from keras.models import load_model
from tkinter import*
from tkinter import ttk
from PIL import Image,ImageTk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import filedialog
from tkinter import messagebox

model=load_model(r"C:\python\Facial Emotion\Facial Emotion\model_file_30epochs.h5")


faceDetect=cv2.CascadeClassifier(r"C:\python\Facial Emotion\Facial Emotion\haarcascade_frontalface_default.xml")

labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

# len(number_of_image), image_height, image_width, channel


root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select image to detect",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
print (root.filename)
#image = cv2.imread(root.filename)
frame=cv2.imread(root.filename)
gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces= faceDetect.detectMultiScale(gray, 1.3, 3)
for x,y,w,h in faces:
    sub_face_img=gray[y:y+h, x:x+w]
    resized=cv2.resize(sub_face_img,(48,48))
    normalize=resized/255.0
    reshaped=np.reshape(normalize, (1, 48, 48, 1))
    result=model.predict(reshaped)
    label=np.argmax(result, axis=1)[0]
    print(label)
    #label = "{}: {:.2f}%".format(label, max(labels_dict) * 100)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
    cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
    cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
cv2.imshow("fer",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()