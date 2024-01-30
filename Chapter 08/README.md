# Chapter 8 Modern Solutions for Object Detection
* https://chtseng.wordpress.com/2018/09/01/%E5%BB%BA%E7%AB%8B%E8%87%AA%E5%B7%B1%E7%9A%84yolo%E8%BE%A8%E8%AD%98%E6%A8%A1%E5%9E%8B-%E4%BB%A5%E6%9F%91%E6%A9%98%E8%BE%A8%E8%AD%98%E7%82%BA%E4%BE%8B/
* https://chtseng.wordpress.com/2018/09/18/%e5%a6%82%e4%bd%95%e4%bc%b0%e7%ae%97%e5%89%8d%e6%96%b9%e4%ba%ba%e7%89%a9%e7%9a%84%e8%b7%9d%e9%9b%a2/

* https://ithelp.ithome.com.tw/articles/10236913
* https://ithelp.ithome.com.tw/articles/10238207
* https://ithelp.ithome.com.tw/articles/10238273
* https://ithelp.ithome.com.tw/articles/10239903
## YOLO 
* Ultralytics official Github repository: https://github.com/ultralytics/ultralytics
* YOLO v5: https://github.com/ultralytics/YOLOv5
* YOLO v7: https://github.com/WongKinYiu/yolov7
* YOLO v8: https://github.com/JustasBart/yolov8_CPP_Inference_OpenCV_ONNX/blob/minimalistic/source/models/yolov8s.onnx

## Open source Visual Servoing Platform
* https://github.com/lagadic/visp
* https://github.com/lagadic/visp/blob/master/tutorial/detection/dnn/coco_classes.txt




## OpenVINO™ Toolkit
### Open Model Zoo repository
* https://github.com/opencv/opencv_zoo

The OpenCV Model Zoo is a valuable resource for computer vision practitioners and researchers. It is a collection of pre-trained models and model weights that can be used with OpenCV. Model Zoo contains a wide range of models designed for different computer vision tasks, including object detection, image classification, face recognition, text detection, and more. These models are trained on large datasets and are often state-of-the-art in terms of accuracy and performance. Many of the models in the OpenCV Model Zoo are compatible with popular deep learning frameworks like TensorFlow, PyTorch, and Caffe.

Using pre-trained models from the Model Zoo is straightforward. OpenCV provides convenient APIs for loading these models and applying them to images or videos. Model Zoo is a collaborative effort, and contributions from the computer vision community are welcome. Developers can find models trained on various datasets and for specific use cases, which can be incredibly valuable for their projects. By leveraging pre-trained models from the Model Zoo, developers can significantly reduce the time and computational resources required for training deep learning models from scratch. Researchers and developers often use these pre-trained models as a starting point for their own experiments or projects. It allows for rapid prototyping and experimentation before committing to training custom models. Whether object recognition, image segmentation, or any other vision-related task, the Model Zoo likely has a pre-trained model that can jumpstart the work.

####  Faster R-CNN with ResNet 50
* Faster R-CNN ResNet-50 model. Used for object detection. For details, see the [paper](https://arxiv.org/abs/1506.01497).
* name: faster_rcnn_resnet50_coco_2018_01_28.tar.gz
* size: 381355771
* checksum: 905eb0e51f1894a02bb8391ed436601450df67e092a9aaee72fc3868d556bbf7410af97a18a06223b389bb818e489999
* original_source: http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz
* source: http://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/faster_rcnn_resnet50_coco/faster_rcnn_resnet50_coco_2018_01_28.tar.gz
* https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/faster_rcnn_resnet50_coco/model.yml
## TensorFlow Object Detection API
* https://github.com/tensorflow/models/tree/master/research/object_detection
 
### MobileNet_SSD_OpenCV_TensorFlow
* github: https://github.com/Qengineering/MobileNet_SSD_OpenCV_TensorFlow
* COCO-trained models: http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz