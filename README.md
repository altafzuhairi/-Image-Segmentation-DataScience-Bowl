#  Cell Nuclei Detection using Semantic Segmantation
# 1. Summary
The aim of this project is to detect cell nuclei using image segmentation process on biomedical images. The images of of the nuclei are acquired varying in cell type, magnification and imaging modality. Deep learning model is trained and deployed for this task. The dataset is acquired from https://www.kaggle.com/competitions/data-science-bowl-2018/data
# 2. IDE and Framework
Spyder is used as the main IDE for this project. The main framework used for this project are TensorFlow, Numpy, Matplotlib, OpenCV and scikit-learn
# 3. Methodology
The methodology for this project is inspired from official TensorFlow website which can be refer here: https://www.tensorflow.org/resources/models-datasets

3.1 Model Pipeline

The model architecture used for this project is U-Net. For further details on the models architecture can be refer on the TensorFlow website. In general, the model consist of a downward stack which serves as a feature extractor. The upward stack produces the pixel-wise output. The model structure is shown in the figure below.

![model](https://user-images.githubusercontent.com/124944787/221126497-c43e3901-b631-4807-9fa3-6f231c28d8ee.png)

The model is trained with batches size 16 and 20 epochs. The training stops at epochs 20 with accuracy of 97% and loss 0.0697.

# 4 .Results

The model is evaluated with test data which is shown in figure below. Some predictions are also made wiht the model using some of the test data. The actual output masks and predicted masks are shown in the figures below.

![show_predictions](https://user-images.githubusercontent.com/124944787/221126787-91c66fee-5c7f-4c19-ae6b-ac565ee73204.png)

Overall the model are able to segmenting the nuclei with excelent accuracy.
