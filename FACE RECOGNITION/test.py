from tensorflow.keras.layers import *
from tensorflow.keras.models import * 
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import os
import win32com.client as wincl
speak = wincl.Dispatch("SAPI.SpVoice")
from time import sleep
#from pygame import mixer

from PIL import Image #Pillow lib for handling images 

#mixer.init()
#sound = mixer.Sound('alarm.wav')



val=0

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

labels = ["user1", "user2", "user3","user4","user5"] 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainner.yml")



count=0
score=0
thicc=2
rpred=[99]
lpred=[99]


def exp(a):
    if(a==0):
        out = "Sudhakar"
        return out
    
    if(a==1):
        out = "Sathish Kumar"
        return out
    
    if(a==2):
        out = "user3"
        return out

    if(a==3):
        out = "user4"
        return out

    if(a==4):
        out = "user5"
        return out

model = load_model("training.h5")
model.summary()
print("success")

#iniciate id counter
id = 0
df=0;
flg=0
flg2=0;
flg3=0;
flg4=0;
flg5=0;

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Vin Diesel', 'Jackie chan', 'raja', 'murugan', 'mano'] 
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img =cam.read()
    #img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        roi_gray = gray[y:y+h, x:x+w] #Convert Face to greyscale 
        id_, conf = recognizer.predict(roi_gray) #recognize the Face
        print(id_)
        print(conf)
        cv2.imwrite("test3.jpg", gray[y:y+h,x:x+w])
        img1 = image.load_img("test3.jpg",target_size=(224,224))
        img1 = image.img_to_array(img1)
        img1 = np.expand_dims(img1,axis=0)   ### flattening
        ypred = model.predict(img1)
        print(ypred)
        
        import numpy
        y = ypred.flatten()
        print('1D Numpy Array:')
        print(y)
        results_index = numpy.argmax(y, axis = 0)
        print(results_index)
        print(y)
        #print(exp(ypred[0]))
        #print(" ")
        font = cv2.FONT_HERSHEY_SIMPLEX
        print(y[results_index])
        # Use putText() method for 
        # inserting text on video
        print(y[results_index] > 0.99)
        
        if conf>=65:
            v=exp(results_index)
        else:
            v="Unknown Person"
        print(v)
        cv2.putText(img, v, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
        if(v=="Sudhakar"):
            flg=flg+1;
            if(flg==30):
                flg=0;
                print("")
                print("Hello Sudhakar, You are logged in, to your account")
                print("")
                speak.Speak("Hello Sudhakar, you are logged in to your account")
                sleep(4)
                while(1):
                    cam.release()
                    cv2.destroyAllWindows()
                    
                    speak.Speak("Please Enter the 4 digit Secret Code")
                    sleep(4)
                    pswd = input("Please Enter the 4 digit Code : ")
                    if(pswd=="1234"):
                        print("Password Matched");
                        speak.Speak("Password Matched")
                        sleep(3)
                        import pandas as pd
                        data = pd.read_csv('bank_dataset.csv')
                        #print(data.head())
                        speak.Speak("Please Enter Amount to with draw")
                        sleep(3)
                        amt = int(input("Please Enter Amount: "))
                        a="Sudhakar"
                        d1 = data.loc[data['Name'] == a]
                        d2 = data.loc[data['Name'] == a].index
                        print(d2)
                        #print(d1)
                        print(d1.index)
                        amount = d1['Total_Amount']
                        print(amount)
                        amount=amount[0]
                        
                        if(amt>amount):
                            df=1
                            speak.Speak("You do not have enough balance to withdraw")
                            print("Your Available Balance is : ", amount)
                            sleep(2)
                            speak.Speak("Thank for using our ATM Service")
                            cam = cv2.VideoCapture(0)
                            cam.set(3, 640) # set video widht
                            cam.set(4, 480) # set video height
                            # Define min window size to be recognized as a face
                            minW = 0.1*cam.get(3)
                            minH = 0.1*cam.get(4)
                        else:
                            df=1
                            amount=amount-amt
                            data.loc[d2,'Total_Amount']=amount
                            print(data)
                            data = data.iloc[:,-5:]
                            data.to_csv("bank_dataset.csv")
                            print("please collect your cash")
                            speak.Speak("Please Collect your cash")
                            sleep(3)
                            print("Your Available Balance is : ", amount)
                            sleep(2)
                            speak.Speak("Thank for using our ATM Service")
                            cam = cv2.VideoCapture(0)
                            cam.set(3, 640) # set video widht
                            cam.set(4, 480) # set video height
                            # Define min window size to be recognized as a face
                            minW = 0.1*cam.get(3)
                            minH = 0.1*cam.get(4)
                            
                    else:
                        print("Password Not Matched");
                        speak.Speak("Password Not Matched")
                        sleep(2)
                        print("Thank You, You are going to log out")
                        sleep(1)
                        cam = cv2.VideoCapture(0)
                        cam.set(3, 640) # set video widht
                        cam.set(4, 480) # set video height
                        # Define min window size to be recognized as a face
                        minW = 0.1*cam.get(3)
                        minH = 0.1*cam.get(4)
                        sleep(3)
                        break
                    if(df==1):
                        df=0
                        break
                        
                        
                id=""
        elif(v=="Sathish Kumar"):
            flg2=flg2+1;
            if(flg2==100):
                flg2=0;
                print("")
                print("Hello Sathish Kumar, You are logged in, to your account")
                print("")
                speak.Speak("Hello Sathish Kumar, you are logged in to your account")
                sleep(4)
                while(1):
                    cam.release()
                    cv2.destroyAllWindows()
                    
                    speak.Speak("Please Enter the 4 digit Secret Code")
                    sleep(4)
                    pswd = input("Please Enter the 4 digit Code : ")
                    if(pswd=="4321"):
                        print("Password Matched");
                        speak.Speak("Password Matched")
                        sleep(3)
                        import pandas as pd
                        data = pd.read_csv('bank_dataset.csv')
                        #print(data.head())
                        speak.Speak("Please Enter Amount to with draw")
                        sleep(3)
                        amt = int(input("Please Enter Amount: "))
                        a="Sathish Kumar"
                        d1 = data.loc[data['Name'] == a]
                        d2 = data.loc[data['Name'] == a].index
                        print(d2)
                        #print(d1)
                        print(d1.index)
                        amount = d1['Total_Amount']
                        amount = amount[1]
                        print(amount)
                        if(amt>amount):
                            df=1
                            speak.Speak("You do not have enough balance to withdraw")
                            print("Your Available Balance is : ", amount)
                            sleep(2)
                            speak.Speak("Thank for using our ATM Service")
                            cam = cv2.VideoCapture(0)
                            cam.set(3, 640) # set video widht
                            cam.set(4, 480) # set video height
                            # Define min window size to be recognized as a face
                            minW = 0.1*cam.get(3)
                            minH = 0.1*cam.get(4)
                        else:
                            amount=amount-amt
                            data.loc[d2,'Total_Amount']=amount
                            print(data)
                            data = data.iloc[:,-5:]
                            data.to_csv("bank_dataset.csv")
                            df=1
                            print("please collect your cash")
                            speak.Speak("Please Collect your cash")
                            sleep(3)
                            print("Your Available Balance is : ", amount)
                            sleep(2)
                            speak.Speak("Thank for using our ATM Service")
                            cam = cv2.VideoCapture(0)
                            cam.set(3, 640) # set video widht
                            cam.set(4, 480) # set video height
                            # Define min window size to be recognized as a face
                            minW = 0.1*cam.get(3)
                            minH = 0.1*cam.get(4)
                            
                    else:
                        print("Password Not Matched");
                        speak.Speak("Password Not Matched")
                        sleep(2)
                        print("Thank You, You are going to log out")
                        sleep(1)
                        cam = cv2.VideoCapture(0)
                        cam.set(3, 640) # set video widht
                        cam.set(4, 480) # set video height
                        # Define min window size to be recognized as a face
                        minW = 0.1*cam.get(3)
                        minH = 0.1*cam.get(4)
                        sleep(3)
                        break
                    if(df==1):
                        df=0
                        break
                        
                        
                id=""
        elif(v=="user3"):
            flg3=flg3+1;
            if(flg3==100):
                flg3=0;
                print("")
                print("Hello user3, You are logged in, to your account")
                print("")
                speak.Speak("Hello user3, you are logged in to your account")
                sleep(4)
                while(1):
                    cam.release()
                    cv2.destroyAllWindows()
                    
                    speak.Speak("Please Enter the 4 digit Secret Code")
                    sleep(4)
                    pswd = input("Please Enter the 4 digit Code : ")
                    if(pswd=="5678"):
                        print("Password Matched");
                        speak.Speak("Password Matched")
                        sleep(3)
                        import pandas as pd
                        data = pd.read_csv('bank_dataset.csv')
                        #print(data.head())
                        speak.Speak("Please Enter Amount to with draw")
                        sleep(3)
                        amt = int(input("Please Enter Amount: "))
                        a="user3"
                        d1 = data.loc[data['Name'] == a]
                        d2 = data.loc[data['Name'] == a].index
                        print(d2)
                        #print(d1)
                        print(d1.index)
                        amount = d1['Total_Amount']
                        amount = amount[2]
                        print(amount)
                        if(amt>amount):
                            df=1
                            speak.Speak("You do not have enough balance to withdraw")
                            print("Your Available Balance is : ", amount)
                            sleep(2)
                            speak.Speak("Thank for using our ATM Service")
                            cam = cv2.VideoCapture(0)
                            cam.set(3, 640) # set video widht
                            cam.set(4, 480) # set video height
                            # Define min window size to be recognized as a face
                            minW = 0.1*cam.get(3)
                            minH = 0.1*cam.get(4)
                        else:
                            amount=amount-amt
                            data.loc[d2,'Total_Amount']=amount
                            data = data.iloc[:,-5:]
                            print(data)
                            data.to_csv("bank_dataset.csv")
                            df=1
                            print("please collect your cash")
                            speak.Speak("Please Collect your cash")
                            sleep(3)
                            print("Your Available Balance is : ", amount)
                            sleep(2)
                            speak.Speak("Thank for using our ATM Service")
                            cam = cv2.VideoCapture(0)
                            cam.set(3, 640) # set video widht
                            cam.set(4, 480) # set video height
                            # Define min window size to be recognized as a face
                            minW = 0.1*cam.get(3)
                            minH = 0.1*cam.get(4)
                            
                    else:
                        print("Password Not Matched");
                        speak.Speak("Password Not Matched")
                        sleep(2)
                        print("Thank You, You are going to log out")
                        sleep(1)
                        cam = cv2.VideoCapture(0)
                        cam.set(3, 640) # set video widht
                        cam.set(4, 480) # set video height
                        # Define min window size to be recognized as a face
                        minW = 0.1*cam.get(3)
                        minH = 0.1*cam.get(4)
                        sleep(3)
                        break
                    if(df==1):
                        df=0
                        break
                        
                        
                id=""
        elif(v=="user4"):
            flg4=flg4+1;
            if(flg4==100):
                flg4=0;
                print("")
                print("Hello user4, You are logged in, to your account")
                print("")
                speak.Speak("Hello user4, you are logged in to your account")
                sleep(4)
                while(1):
                    cam.release()
                    cv2.destroyAllWindows()
                    
                    speak.Speak("Please Enter the 4 digit Secret Code")
                    sleep(4)
                    pswd = input("Please Enter the 4 digit Code : ")
                    if(pswd=="8765"):
                        print("Password Matched");
                        speak.Speak("Password Matched")
                        sleep(3)
                        import pandas as pd
                        data = pd.read_csv('bank_dataset.csv')
                        #print(data.head())
                        speak.Speak("Please Enter Amount to with draw")
                        sleep(3)
                        amt = int(input("Please Enter Amount: "))
                        a="user4"
                        d1 = data.loc[data['Name'] == a]
                        d2 = data.loc[data['Name'] == a].index
                        print(d2)
                        #print(d1)
                        print(d1.index)
                        amount = d1['Total_Amount']
                        print(amount)
                        amount=amount[3]
                        if(amt>amount):
                            df=1
                            speak.Speak("You do not have enough balance to withdraw")
                            print("Your Available Balance is : ", amount)
                            sleep(2)
                            speak.Speak("Thank for using our ATM Service")
                            cam = cv2.VideoCapture(0)
                            cam.set(3, 640) # set video widht
                            cam.set(4, 480) # set video height
                            # Define min window size to be recognized as a face
                            minW = 0.1*cam.get(3)
                            minH = 0.1*cam.get(4)
                        else:
                            amount=amount-amt
                            data.loc[d2,'Total_Amount']=amount
                            print(data)
                            data = data.iloc[:,-5:]
                            data.to_csv("bank_dataset.csv")
                            df=1
                            print("please collect your cash")
                            speak.Speak("Please Collect your cash")
                            sleep(3)
                            print("Your Available Balance is : ", amount)
                            sleep(2)
                            speak.Speak("Thank for using our ATM Service")
                            cam = cv2.VideoCapture(0)
                            cam.set(3, 640) # set video widht
                            cam.set(4, 480) # set video height
                            # Define min window size to be recognized as a face
                            minW = 0.1*cam.get(3)
                            minH = 0.1*cam.get(4)
                            
                    else:
                        print("Password Not Matched");
                        speak.Speak("Password Not Matched")
                        sleep(2)
                        print("Thank You, You are going to log out")
                        sleep(1)
                        cam = cv2.VideoCapture(0)
                        cam.set(3, 640) # set video widht
                        cam.set(4, 480) # set video height
                        # Define min window size to be recognized as a face
                        minW = 0.1*cam.get(3)
                        minH = 0.1*cam.get(4)
                        sleep(3)
                        break
                    if(df==1):
                        df=0
                        break
                        
                        
                id=""
        elif(v=="user5"):
            flg5=flg5+1;
            if(flg5==100):
                flg5=0;
                print("")
                print("Hello user5, You are logged in, to your account")
                print("")
                speak.Speak("Hello user5, you are logged in to your account")
                sleep(4)
                while(1):
                    cam.release()
                    cv2.destroyAllWindows()
                    
                    speak.Speak("Please Enter the 4 digit Secret Code")
                    sleep(4)
                    pswd = input("Please Enter the 4 digit Code : ")
                    if(pswd=="7654"):
                        print("Password Matched");
                        speak.Speak("Password Matched")
                        sleep(3)
                        import pandas as pd
                        data = pd.read_csv('bank_dataset.csv')
                        #print(data.head())
                        speak.Speak("Please Enter Amount to with draw")
                        sleep(3)
                        amt = int(input("Please Enter Amount: "))
                        a="Priya"
                        d1 = data.loc[data['Name'] == a]
                        d2 = data.loc[data['Name'] == a].index
                        print(d2)
                        #print(d1)
                        print(d1.index)
                        amount = d1['Total_Amount']
                        print(amount)
                        amount=amount[4]
                        if(amt>amount):
                            df=1
                            speak.Speak("You do not have enough balance to withdraw")
                            print("Your Available Balance is : ", amount)
                            sleep(2)
                            speak.Speak("Thank for using our ATM Service")
                            cam = cv2.VideoCapture(0)
                            cam.set(3, 640) # set video widht
                            cam.set(4, 480) # set video height
                            # Define min window size to be recognized as a face
                            minW = 0.1*cam.get(3)
                            minH = 0.1*cam.get(4)
                        else:
                            amount=amount-amt
                            data.loc[d2,'Total_Amount']=amount
                            data = data.iloc[:,-5:]
                            print(data)
                            data.to_csv("bank_dataset.csv")
                            df=1
                            print("please collect your cash")
                            speak.Speak("Please Collect your cash")
                            sleep(3)
                            print("Your Available Balance is : ", amount)
                            sleep(2)
                            speak.Speak("Thank for using our ATM Service")
                            cam = cv2.VideoCapture(0)
                            cam.set(3, 640) # set video widht
                            cam.set(4, 480) # set video height
                            # Define min window size to be recognized as a face
                            minW = 0.1*cam.get(3)
                            minH = 0.1*cam.get(4)
                            
                    else:
                        print("Password Not Matched");
                        speak.Speak("Password Not Matched")
                        sleep(2)
                        print("Thank You, You are going to log out")
                        sleep(1)
                        cam = cv2.VideoCapture(0)
                        cam.set(3, 640) # set video widht
                        cam.set(4, 480) # set video height
                        # Define min window size to be recognized as a face
                        minW = 0.1*cam.get(3)
                        minH = 0.1*cam.get(4)
                        sleep(3)
                        break
                    if(df==1):
                        df=0
                        break
                        
                        
                id=""

    cv2.imshow('camera',img)
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n Program Ended")
cam.release()
cv2.destroyAllWindows()
