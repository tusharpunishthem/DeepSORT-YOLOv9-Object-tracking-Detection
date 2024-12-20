<H1 align="center">
YOLOv9 Object Detection with DeepSORT Tracking(ID + Trails) for Both CPU & GPU </H1>

### New Features
- Added Label for Every Track
- Code can run on Both (CPU & GPU)
- Video/WebCam/External Camera/IP Stream Supported


## Steps to run Code

- Clone the repository
```
git clone https://github.com/tusharpunishthem/deepsortyolov9obj.git
```
- Goto the cloned folder.
```
cd deepsortyolov9obj
```
Create conda environment Python 3.10
```
conda create -n yolov9_tracking python=3.10
conda activate yolov9_tracking
```
- Install requirements with mentioned command below.
```
pip install -r requirements.txt
```
- Download the pre-trained YOLOv9 model weights
[yolov9](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt)

- Downloading the DeepSORT Files From The Google Drive 
```
gdown "https://drive.google.com/uc?id=11ZSZcG-bcbueXZC3rN08CM0qqX3eiHxf&confirm=t"
```
- After downloading the DeepSORT Zip file from the drive, unzip it. 

- Download sample videos from the Google Drive
```
gdown "https://drive.google.com/uc?id=115RBSjNQ_1zjvKFRsQK2zE8v8BIRrpdy&confirm=t"
gdown "https://drive.google.com/uc?id=1rjBn8Fl1E_9d0EMVtL24S9aNQOJAveR5&confirm=t"
```

```
Check if cuda is available or not for GPU (ensure that you got NVIDIA GPU)
1.Type "python" in anaconda environment
2.import torch

print("Number of CUDA Devices:", torch.cuda.device_count())
print("Current CUDA Device:", torch.cuda.current_device())

Output: if the device turn out to be zero Follow below steps -
Install Nvidia Cuda toolkit - "https://developer.nvidia.com/cuda-12-0-0-download-archive"
Now check cuda version : nvcc --version
In your conda prompt type: device = torch.device("cuda:0")  # Use the first GPU
conda install cuda --channel nvidia/label/cuda-12.0
cuda will be installed!
```



# for detection only
python detect_dual.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0 --view-img

#for detection and tracking
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0 --view-img

#for WebCam
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 0 --device 0 --view-img

#for External Camera
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 1 --device 0

#For LiveStream (Ip Stream URL Format i.e "rtsp://username:pass@ipaddress:portno/video/video.amp")
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source "your IP Camera Stream URL" --device 0 --view-img

#for specific class (person)
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0 --classes 0 --view-img

#for detection and tracking with trails 
!python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0 --draw-trails --view-img
```

- Output file will be created in the ```working-dir/runs/detect/obj-tracking``` with original filename








## References
- [Implementation of paper - YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://github.com/WongKinYiu/yolov9/blob/main/README.md)
- [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)
- [YOLOv9 Object Detection with DeepSORT Tracking](https://github.com/MuhammadMoinFaisal/YOLOv9-DeepSORT-Object-Tracking.git)
