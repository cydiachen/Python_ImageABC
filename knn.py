# Basic Ideas：
# Step#1: Gather our dataset : resize to 32*32 *rgb(3 channels) 收集我们的数据集，并且把每一张图片设置称为32*32的rgb图片
# Step#2: Split dataset: training set and test set  把我们的数据集分割成训练集和测试集
# Step#3: Train the classifier: 直接训练我们的分类器
# Step#4: Evaluate: Test on performance 对于我们系统的性能进行一个测试

# Import Necessary packages: 引入我们需要的包

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from Python_ImageABC.preprocessing import SimplePreprocessor
from Python_ImageABC.preprocessing import SimpleDatasetLoader
from imutils import paths
import argparse

