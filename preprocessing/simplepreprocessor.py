import cv2

class SimplePreprocessor:
    def __init__(self,width,height,inter=cv2.INTER_AREA):
        #保存这些信息的目的很简单，很多时候我们是需要进行resize的操作的，这个时候我们保存的width，height和inter就非常有用了。
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self,image):
        #这里的这些方法的话就非常的假单，我们只需要按照我们需要的图片的长、宽像素值进行赋值就行，这里面的话是不考虑图像原生比例的
        return cv2.resize(image,(self.width,self.height),interpolation=self.inter)

