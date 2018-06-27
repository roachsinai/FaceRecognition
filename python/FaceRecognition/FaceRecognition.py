import numpy as np 
import cv2
import face_recognition as fr
import threading
import time
import yaml
def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat
yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)
class FaceRecognition(object):
    def __init__(self):
        self.face_locations=[]
        self.face_encodings=[]
        self.face_names=[]

    def find_lanmark_recognize_face(self):
        dir="F:/Datasets/LFW/lfw-deepfunneled/Aleksander_Kwasniewski/"
        name="Aleksander_Kwasniewski_0001.jpg"
        path=dir+name
        img=fr.load_image_file(path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        face_locations=fr.face_locations(img)
        face_landmarks_list=fr.face_landmarks(img,face_locations)
        detected_num=len(face_locations)
        if 0:
            for i in range(detected_num):
                img1=img[face_locations[i][0]:face_locations[i][2],face_locations[i][3]:face_locations[i][1],:]
                cv2.namedWindow('x1',0)
                cv2.imshow('x1',img1)
                cv2.waitKey(0)
        if 0:
            facial_features=[
                'bottom_lip',
                'chin',
                'left_eye',
                'left_eyebrow',
                'nose_bridge',
                'nose_tip',
                'right_eye',
                'right_eyebrow',
                'top_lip'
                ]
            for face_landmarks in face_landmarks_list:
                for facial_feature in facial_features:
                    #if facial_feature=='bottom_lip':
                    #    continue
                    feature_size=len(face_landmarks[facial_feature])
                    for i in range(feature_size-1):
                        cv2.line(img,face_landmarks[facial_feature][i],face_landmarks[facial_feature][i+1],(255,255,255),1)
        if 0:
            dir="F:/Datasets/LFW/lfw-deepfunneled/  Aleksander_Kwasniewski/"
            name="Aleksander_Kwasniewski_0001.jpg"
            name1="Aleksander_Kwasniewski_0002.jpg"
            path=dir+name
            path1=dir+name1
            known_img=fr.load_image_file(path)
            unknown_img=fr.load_image_file(path1)
            known_encoding=fr.face_encodings(known_img)[0]
            unknown_encoding=fr.face_encodings(unknown_img) [0]
            res=fr.compare_faces    ([known_encoding],unknown_encoding)
            print(res)
        cv2.namedWindow('x',0)
        cv2.imshow('x',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pass

    def analyze_frame(self):
        while True:
            time.sleep(0.001)
            self.face_locations=[]
            self.face_encodings=[]
            self.face_names=[]
            frame=self.frame
            if frame is None:
                continue
            self.face_locations=fr.face_locations(frame)
            self.face_encodings=fr.face_encodings(frame,self.face_locations)
            for face_encoding in self.face_encodings:
                matches=fr.compare_faces(self.known_face_encodings,face_encoding)
                name="Unknown"
                if True in matches:
                    first_match_idx=matches.index(True)
                    name=self.known_face_name[first_match_idx]
                self.face_names.append(name)
            pass

    def recognition_from_camera(self):
        capture=cv2.VideoCapture(0)
        self.known_face_encodings=[]
        known_face_name=[]
        self.known_face_encodings,self.known_face_name=self.face_database()

        times=0
        #t1=threading.Thread( target=self.analyze_frame )
        #t1.start()
        self.face_locations
        self.face_encodings
        self.face_names
        face_landmarks_list=[]
        facial_features=[
            'bottom_lip',
            'chin',
            'left_eye',
            'left_eyebrow',
            'nose_bridge',
            'nose_tip',
            'right_eye',
            'right_eyebrow',
            'top_lip'
            ]
        #filename = "./../images/CameraCalibration/out_camera_data_left.yml";
        #with open(filename) as fin:
        #    calibrated_data = yaml.load(fin.read())
        #cameraMatrix
        #distCoeffs
        while True:
            times+=1
            ret,self.frame=capture.read()
            frame=self.frame
            #small_face=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)

            #tmp=frame.copy()
            #frame=cv2.undistort(temp, cameraMatrix, distCoeffs)

            if times%2==0:
                self.face_locations=[]
                self.face_encodings=[]
                self.face_names=[]
                self.face_locations=fr.face_locations(frame)
                face_landmarks_list=fr.face_landmarks(frame,self.face_locations)

                self.face_encodings=fr.face_encodings(frame,self.face_locations)
                for face_encoding in self.face_encodings:
                    matches=fr.compare_faces(self.known_face_encodings,face_encoding)
                    name="Unknown"
                    if True in matches:
                        first_match_idx=matches.index(True)
                        name=self.known_face_name[first_match_idx]
                    self.face_names.append(name)

            for face_landmarks in face_landmarks_list:
                for facial_feature in facial_features:
                    feature_size=len(face_landmarks[facial_feature])
                    for i in range(feature_size-1):
                        cv2.line(frame,face_landmarks[facial_feature][i],face_landmarks[facial_feature][i+1],(255,255,255),1)

            for (top,right,bottom,left),name in zip(self.face_locations,self.face_names):
                cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
                cv2.rectangle(frame,(left,bottom),(right,bottom+15),(0,0,255),cv2.FILLED)
                font=cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame,name,(left+6,bottom+11),font,0.3,(255,255,255),1)
            cv2.namedWindow('x',0)
            cv2.imshow('x',self.frame)
            cv2.waitKey(1)
        capture.release()
        cv2.destroyAllWindow()

    def face_database(self):
        dir="F:/images/People/"
        name="Wu_Yanzu_1.jpg"
        name1="Wu_Yanzu_2.jpg"
        name2="Wu_Yanzu_3.jpg"
        path=dir+name
        path1=dir+name1
        path2=dir+name2
        img=fr.load_image_file(path)
        img1=fr.load_image_file(path1)
        img2=fr.load_image_file(path2)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

        face_location=fr.face_locations(img)
        face_encoding=fr.face_encodings(img,face_location)[0]
        face_location=fr.face_locations(img1)
        face_encoding1=fr.face_encodings(img,face_location)[0]
        face_location=fr.face_locations(img2)
        face_encoding2=fr.face_encodings(img,face_location)[0]
        known_face_encodings=[
            face_encoding,
            face_encoding1,
            face_encoding2,
            ]
        known_face_name=[
            "Wu Yanzu",
            "Wu Yanzu",
            "Wu Yanzu",
            ]
        return known_face_encodings,known_face_name

def main(): 
    model= FaceRecognition()
    #model.find_lanmark_recognize_face()
    model.recognition_from_camera()

if __name__ == '__main__':
    main()