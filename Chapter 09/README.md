# Chapter 9 Faces and Text
## Key terms
* Face detection: A computer vision use case to determine presence of human faces in an image.
* Haar cascades: An algorithm for rapid detection of objects in an image.
* Viola Jones algorithm: Algorithm that uses Haar cascades for face detection.
* YuNet: A deep learning algorithm built using Mobilenet for face detection.
* Facial Landmarks: Key points of a human face used for various computer vision tasks.
* LFW dataset: Labeled Faces in the Wild dataset containing famces of celebrities under various lighting conditions.
* Text detection: A computer vision use case to determine presence of text in an image.
* Text recognition: A computer vision use case to identify the text alphabets, numbers, special symbols in the image.
* OCR: Optical Character Recognition. Another name for text recognition.
* OpenCV Model Zoo: An opensource repository of popular models for solving computer vision usecases.

## Open Model Zoo repository
* https://github.com/opencv/opencv_zoo

The OpenCV Model Zoo is a valuable resource for computer vision practitioners and researchers. It is a collection of pre-trained models and model weights that can be used with OpenCV. Model Zoo contains a wide range of models designed for different computer vision tasks, including object detection, image classification, face recognition, text detection, and more. These models are trained on large datasets and are often state-of-the-art in terms of accuracy and performance. Many of the models in the OpenCV Model Zoo are compatible with popular deep learning frameworks like TensorFlow, PyTorch, and Caffe.

Using pre-trained models from the Model Zoo is straightforward. OpenCV provides convenient APIs for loading these models and applying them to images or videos. Model Zoo is a collaborative effort, and contributions from the computer vision community are welcome. Developers can find models trained on various datasets and for specific use cases, which can be incredibly valuable for their projects. By leveraging pre-trained models from the Model Zoo, developers can significantly reduce the time and computational resources required for training deep learning models from scratch. Researchers and developers often use these pre-trained models as a starting point for their own experiments or projects. It allows for rapid prototyping and experimentation before committing to training custom models. Whether object recognition, image segmentation, or any other vision-related task, the Model Zoo likely has a pre-trained model that can jumpstart the work.

### Haar cascades algorithm
   haarcascade_frontalface_default.xml
* **face_detection.py**: opencv 

### YuNet
* github: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
* **face_detection_dnn.py**: face_detection_yunet_2023mar.onnx

### SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition
* github: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
* **face_recognition.py**: face_detection_yunet_2023mar.onnx

## Real-time Scene Text Detection with Differentiable Binarization
* gitcode: https://gitcode.net/opencv/opencv_zoo/-/tree/aaa0d19b39a3096616978741e71d16c634a51b69/models/text_detection_db
* **text_detection.py**: text_detection_DB_TD500_resnet18_2021sep.onnx

## CRNN Text Recognition
* github: https://github.com/opencv/opencv_zoo/tree/main/models/text_recognition_crnn
* gitcode: https://gitcode.net/opencv/opencv_zoo/-/tree/master/models/text_recognition_crnn
* **text_recognition.py**: text_recognition_CRNN_EN_2021sep.onnx
