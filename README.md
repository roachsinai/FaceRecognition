# FaceRecognition
My small step into the face recognition field

I will describe how to handle this problem from scratch as following:

1. I searched some public resources about face recognition field， then I found https://www.youtube.com/watch?v=LYgBqJorF44&t=2324s.
   I have learned EigenFaces and FisherFaces basic concepts from it. 
2. I found https://github.com/ageitgey/face_recognition, then implement this API in my code.
3. I searched for face datasets, then I have found LFW. I read all stuffs in it's website like Results and Errata.
4. I coded EigenFaces by myself using C++, then applied to LFW. I do many predictions to feel characristics of this method.
5. I built a neural network model based on SqueezeNet using python, then applied to LFW. I trained 1000 images 
   and reached 50% accuracy within 20 epochs, and batch size is 100.
6. I tried to code little FisherFaces and Adaboost example following some snippets from website I can find.
7. I use OpenCV FileStorage to save data and share data between C++ and python. My machine can handle 3000 data every time as 
   maximum(but it is too slow, so I use 1000 images to train).

Becaue time limits, I do not choose read some systematic books or read professional papers about face recognition. 
I have much work to do every day. But I am able to read systematic books or read papers for the further research.
I am confident I will go further as time goes by. And these steps I want to prove I am good at coding, I have good foundation at math 
especially Algebra. 
