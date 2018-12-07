# Basic Ideas：
# Step#1: Gather our dataset : resize to 32*32 *rgb(3 channels) 收集我们的数据集，并且把每一张图片设置称为32*32的rgb图片
# Step#2: Split dataset: training set and test set  把我们的数据集分割成训练集和测试集
# Step#3: Train the classifier: 直接训练我们的分类器
# Step#4: Evaluate: Test on performance 对于我们系统的性能进行一个测试

# Import Necessary packages: 引入我们需要的包
# 这里的话需要注意一个问题，因为这里的话我对于import包的方法还不太熟练，这里为了不浪费时间，目前的Alpha版本，仅仅是把我们的几个函数直接引进来

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from imutils import paths
import argparse

#================下面的部分的话是Simple Dataset Loader的部分============================
import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self,preprocessors = None):
        # 这里的话我们需要保存我们的图像的预处理
        self.preprocessors = preprocessors

        #这里的话，如果我们的预处理工作没有做的话，我们需要把这个预处理的过程作为一个空的列表
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self,imagePaths, verbose = -1):
        #这里我们先对于我们的features和labels进行初始化
        data = []
        labels = []

        #接着的话我们需要循环得到我们所有的输入图像
        for (i, imagePath) in enumerate (imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            
            if self.preprocessors is not None:
                #这里的话，因为我们需要做的预处理过程，那么我们就需要把每一个预处理读进来
                for p in self.preprocessors:
                    image = p.preprocess(image)
            
            #这里的话，我们把我们的processed image作为一个feature vector
            #我们需要通过图片的label来把数据更新到我们的list中间
            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i+1)% verbose == 0:
                print('[INFO] Processed: {}/{}'.format(i+1,len(imagePaths)))

            #这里的话我们需要return一个data和label的元祖
            return (np.array(data),np.array(labels))

#================至此为止==============================


#================下面的部分的话是Simple Preprocessor的部分============================
class SimplePreprocessor:
    def __init__(self,width,height,inter=cv2.INTER_AREA):
        #保存这些信息的目的很简单，很多时候我们是需要进行resize的操作的，这个时候我们保存的width，height和inter就非常有用了。
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self,image):
        #这里的这些方法的话就非常的假单，我们只需要按照我们需要的图片的长、宽像素值进行赋值就行，这里面的话是不考虑图像原生比例的
        return cv2.resize(image,(self.width,self.height),interpolation=self.inter)
#=================至此为止============================


#下面的话，我们就需要构建我们的参数parser
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required = True, help = "Path to input dataset")
ap.add_argument("-k","--neighbors",type = int, default=1,help="# of nearest neighbor for classification")
ap.add_argument("-j","--jobs",type = int, default=-1, help = "# of jobs for K-NN distance(-1 uses all available cores)")
args = vars(ap.parse_args())

#这个时候，我们的内容都可以被解析了，这里的话我们就可以获得图片的文件地址
'''
--dataset: 我们输入图像的数据集
--neighbors: 可选的KNN算法的比较点书
--jobs: 计算距离时要运行的并发作业数在输入数据点和训练集之间。 -1代表将使用所有可用核心处理器。
'''

print("[INFO] LOADING IMAGES...")
imagePaths = list(paths.list_images(args["dataset"]))

# 这里的话我们需要初始化我们的image preprocessor，从本地磁盘读取dataset，同时的话对于我们的data matrix进行reshape
sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data,labels) = sdl.load(imagePaths,verbose = 500)
data =  data.reshape((data.shape[0],3072))

#这里的话我们需要显示一下图片占用的内存信息量
print("[INFO] features matrix:{:.1f}MB".format(data.nbytes / (1024 * 1000.0)))



#这里的话，我们需要把我们的文字标签转换为整数
le = LabelEncoder()
labels = le.fit_transform(labels)

#这里的话，我们需要把读取出来的数据分割称为75%的训练集以及25%的测试机
(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size = 0.25,random_state = 42)

# 下面的话我们就要训练我们的K-means classifier
print("[INFO] evaluating K-NN classifier...")
model = KNeighborsClassifier(n_neighbors = args["neighbors"],n_jobs = args["jobs"])
model.fit(trainX,trainY)
print(classification_report(testY,model.predict(testX),target_names = le.classes_))

