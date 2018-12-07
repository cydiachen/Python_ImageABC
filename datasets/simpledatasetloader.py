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

        # 这里的话，我们的需要的dataset的格式是：/dataset_name/class/image.jpg
        # 如果我们使用别人的数据集的时候，我们就需要对于这些格式进行自己的定制
        # 在这个例子里面，我们使用的每一个class是别人已经进行了分类好的图像，因为这里的话imagepath相当于是一个相对路径的雄狮/dataset_name/class/
        # 这里的话就涉及到我们如何去正确的组织整理我们的数据集了，这个也是一个我们目前可以做的工作




