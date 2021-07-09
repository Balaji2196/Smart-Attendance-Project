import cv2
import numpy as np
import face_recognition
import os
import smtplib
from datetime import datetime


time_new = datetime.now()
date_name = time_new.strftime('%d/%m/%Y')
time_name = time_new.strftime('%H:%M:%S')
date_s=date_name.split('/')

file="{}_{}_{}.csv".format(date_s[0],date_s[1],date_s[2])

open(file,'a').close()
    


path = 'image_p'
images = []
personNames = []
myList = os.listdir(path)

for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def send_mail(to_mail,message,time,date,name):
    
    server=smtplib.SMTP_SSL("smtp.gmail.com",465)
    server.login("ENTER YOUR EMAIL ID HERE","ENTER YOUR PASSWORD HERE")
    
    server.sendmail("ENTER YOUR EMAIL ID HERE",to_mail,message)             
    print("mail is sent to {} for {}'s presence".format(to_mail,name))
    server.quit()
    
            


def attendance(name):
    file_date=datetime.now()
    date_name_new = file_date.strftime('%d/%m/%Y')
    file_date=date_name_new.split('/')
    file_gen="{}_{}_{}.csv".format(file_date[0],file_date[1],file_date[2])
    with open(file_gen, 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        to_mail=""
        #person_list=['hrithik roshan', 'hugh jackman', 'deepika padukone', 'salman khan', 'sharukh khan'] => names should be same like the images in image_p folder
        person_list=personNames
        #enter the mail id's which you want to get mail in the below personNames_parent_mail list
        personNames_parent_mail=["user1@gmail.com","user2@gmail.com","user3@gmail.com","user4@gmail.com","user5@gmail.com"]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')
            message="\n{} is present- Entry Time: {}   Date: {}  ".format(name,tStr,dStr)
            print(message)
            print(name)
            name=name.lower()
            if name == person_list[0].lower():
                to_mail=personNames_parent_mail[0]
            if name == person_list[1].lower():
                to_mail=personNames_parent_mail[1]
            if name == person_list[2].lower():
                to_mail=personNames_parent_mail[2]
            if name == person_list[3].lower():
                to_mail=personNames_parent_mail[3]
            if name == person_list[4].lower():
                to_mail=personNames_parent_mail[4]
            send_mail(to_mail,message,tStr,dStr,name)
            
            


encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,150)


thresh=0.5

config_path='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_path='frozen_inference_graph.pb'
classfile='coco.names'
classnames=[]
with open(classfile,'rt') as f:
    names=f.read().splitlines()
print(names)

net=cv2.dnn_DetectionModel(weights_path,config_path)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

mask_list=['cell phone','laptop','tv']


while True:
    ret, frame = cap.read()
    
    classIds,confs,bbox=net.detect(frame,confThreshold=thresh)
    if len(classIds)!=0:
      for (classId,confidence,box) in zip(classIds.flatten(),confs.flatten(),bbox):
        
        if names[classId-1] in mask_list:
            cv2.rectangle(frame,box,(0,255,0),-1)
            #cv2.putText(frame, names[classId - 1].upper(), (box[0] + 10, box[1] + 30),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            #cv2.putText(frame,str(round(confidence*100)),(box[0]+300,box[1]+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


 
