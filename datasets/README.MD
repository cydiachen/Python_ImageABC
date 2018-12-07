# 简单的Data Loader程序

# 0.0 简介

这里的话，我们的simpledatasetloader是非常有必要存在的。

这里我们使用一个preprocessor作为一个列表而不是单独值的形式，这个技巧是非常重要的。因为我们往往在操作的时候我们需要调整图像的尺寸啊，然后使用一些scaling方法，比如说我们可能要用mean subtraction。有的时候，我们在训练的时候还有可能需要对于我们的图片进行augmentation操作，这里的话就需要我们仔细的进行处理。

# 1.0 参数意义

# 2.0 注意事项

在我们的simple的dataset loader之中，我们的文件存储必须是以这种形式：
/dataset_name/class/image.jpg

比如说：
/CAT&DOG/cat/image_01.jpg 这种形式才可以

