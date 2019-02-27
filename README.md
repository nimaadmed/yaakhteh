# yaakhteh


This is a primary version of software which can segment the peripheral blood cells and diagnosis them.\
For segmentation part we use [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rc).\
For detection part we use pretrained [ResNet18](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py).

## Requirements
This software is developed in python3 and you should install following libraries:
-pytorch\
-tensorflow\
-numpy\
-PIL\
-cv2\
-torchvision\
-tkinter

## Installation
1. Clone the repository
```
https://github.com/nimaadmed/yaakhteh
```
2. Build tf-faster-rcnn part 
```
cd Cell_detection\lib
make clean
make
cd ..

```
3. Download models and put them in models folder
4. Run the Detection code:
```
python3 tools/cell_detection.py
```

## Result
You can see an example output of software in following picture:
![](https://github.com/nimaadmed/yaakhteh/blob/master/result.png)







