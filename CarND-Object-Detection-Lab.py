#!/usr/bin/env python
# coding: utf-8

# # CarND Object Detection Lab
# 
# Let's get started!

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import time
from scipy.stats import norm

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# ## MobileNets
# 
# [*MobileNets*](https://arxiv.org/abs/1704.04861), as the name suggests, are neural networks constructed for the purpose of running very efficiently (high FPS, low memory footprint) on mobile and embedded devices. *MobileNets* achieve this with 3 techniques:
# 
# 1. Perform a depthwise convolution followed by a 1x1 convolution rather than a standard convolution. The 1x1 convolution is called a pointwise convolution if it's following a depthwise convolution. The combination of a depthwise convolution followed by a pointwise convolution is sometimes called a separable depthwise convolution.
# 2. Use a "width multiplier" - reduces the size of the input/output channels, set to a value between 0 and 1.
# 3. Use a "resolution multiplier" - reduces the size of the original input, set to a value between 0 and 1.
# 
# These 3 techniques reduce the size of cummulative parameters and therefore the computation required. Of course, generally models with more paramters achieve a higher accuracy. *MobileNets* are no silver bullet, while they perform very well larger models will outperform them. ** *MobileNets* are designed for mobile devices, NOT cloud GPUs**. The reason we're using them in this lab is automotive hardware is closer to mobile or embedded devices than beefy cloud GPUs.

# ### Convolutions
# 
# #### Vanilla Convolution
# 
# Before we get into the *MobileNet* convolution block let's take a step back and recall the computational cost of a vanilla convolution. There are $N$ kernels of size $D_k * D_k$. Each of these kernels goes over the entire input which is a $D_f * D_f * M$ sized feature map or tensor (if that makes more sense). The computational cost is:
# 
# $$
# D_g * D_g * M * N * D_k * D_k
# $$
# 
# Let $D_g * D_g$ be the size of the output feature map. Then a standard convolution takes in a $D_f * D_f * M$ input feature map and returns a $D_g * D_g * N$ feature map as output.
# 
# (*Note*: In the MobileNets paper, you may notice the above equation for computational cost uses $D_f$ instead of $D_g$. In the paper, they assume the output and input are the same spatial dimensions due to stride of 1 and padding, so doing so does not make a difference, but this would want $D_g$ for different dimensions of input and output.)
# 
# ![Standard Convolution](assets/standard_conv.png)
# 
# 
# 
# #### Depthwise Convolution
# 
# A depthwise convolution acts on each input channel separately with a different kernel. $M$ input channels implies there are $M$ $D_k * D_k$ kernels. Also notice this results in $N$ being set to 1. If this doesn't make sense, think about the shape a kernel would have to be to act upon an individual channel.
# 
# Computation cost:
# 
# $$
# D_g * D_g * M * D_k * D_k
# $$
# 
# 
# ![Depthwise Convolution](assets/depthwise_conv.png)
# 
# 
# #### Pointwise Convolution
# 
# A pointwise convolution performs a 1x1 convolution, it's the same as a vanilla convolution except the kernel size is $1 * 1$.
# 
# Computation cost:
# 
# $$
# D_k * D_k * D_g * D_g * M * N =
# 1 * 1 * D_g * D_g * M * N =
# D_g * D_g * M * N
# $$
# 
# ![Pointwise Convolution](assets/pointwise_conv.png)
# 
# 
# 
# Thus the total computation cost is for separable depthwise convolution:
# 
# $$
# D_g * D_g * M * D_k * D_k + D_g * D_g * M * N
# $$
# 
# which results in $\frac{1}{N} + \frac{1}{D_k^2}$ reduction in computation:
# 
# $$
# \frac {D_g * D_g * M * D_k * D_k + D_g * D_g * M * N} {D_g * D_g * M * N * D_k * D_k} = 
# \frac {D_k^2 + N} {D_k^2*N} = 
# \frac {1}{N} + \frac{1}{D_k^2}
# $$
# 
# *MobileNets* use a 3x3 kernel, so assuming a large enough $N$, separable depthwise convnets are ~9x more computationally efficient than vanilla convolutions!

# ### Width Multiplier
# 
# The 2nd technique for reducing the computational cost is the "width multiplier" which is a hyperparameter inhabiting the range [0, 1] denoted here as $\alpha$. $\alpha$ reduces the number of input and output channels proportionally:
# 
# $$
# D_f * D_f * \alpha M * D_k * D_k + D_f * D_f * \alpha M * \alpha N
# $$

# ### Resolution Multiplier
# 
# The 3rd technique for reducing the computational cost is the "resolution multiplier" which is a hyperparameter inhabiting the range [0, 1] denoted here as $\rho$. $\rho$ reduces the size of the input feature map:
# 
# $$
# \rho D_f * \rho D_f * M * D_k * D_k + \rho D_f * \rho D_f * M * N
# $$

# Combining the width and resolution multipliers results in a computational cost of:
# 
# $$
# \rho D_f * \rho D_f * a M * D_k * D_k + \rho D_f * \rho D_f * a M * a N
# $$
# 
# Training *MobileNets* with different values of $\alpha$ and $\rho$ will result in different speed vs. accuracy tradeoffs. The folks at Google have run these experiments, the result are shown in the graphic below:
# 
# ![MobileNets Graphic](https://github.com/tensorflow/models/raw/master/research/slim/nets/mobilenet_v1.png)

# MACs (M) represents the number of multiplication-add operations in the millions.

# ### Exercise 1 - Implement Separable Depthwise Convolution
# 
# In this exercise you'll implement a separable depthwise convolution block and compare the number of parameters to a standard convolution block. For this exercise we'll assume the width and resolution multipliers are set to 1.
# 
# Docs:
# 
# * [depthwise convolution](https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d)

# In[ ]:


def vanilla_conv_block(x, kernel_size, output_channels):
    """
    Vanilla Conv -> Batch Norm -> ReLU
    """
    x = tf.layers.conv2d(
        x, output_channels, kernel_size, (2, 2), padding='SAME')
    x = tf.layers.batch_normalization(x)
    return tf.nn.relu(x)

# TODO: implement MobileNet conv block
def mobilenet_conv_block(x, kernel_size, output_channels):
    """
    Depthwise Conv -> Batch Norm -> ReLU -> Pointwise Conv -> Batch Norm -> ReLU
    """
    pass


# **[Sample solution](./exercise-solutions/e1.py)**
# 
# Let's compare the number of parameters in each block.

# In[ ]:


# constants but you can change them so I guess they're not so constant :)
INPUT_CHANNELS = 32
OUTPUT_CHANNELS = 512
KERNEL_SIZE = 3
IMG_HEIGHT = 256
IMG_WIDTH = 256

with tf.Session(graph=tf.Graph()) as sess:
    # input
    x = tf.constant(np.random.randn(1, IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS), dtype=tf.float32)

    with tf.variable_scope('vanilla'):
        vanilla_conv = vanilla_conv_block(x, KERNEL_SIZE, OUTPUT_CHANNELS)
    with tf.variable_scope('mobile'):
        mobilenet_conv = mobilenet_conv_block(x, KERNEL_SIZE, OUTPUT_CHANNELS)

    vanilla_params = [
        (v.name, np.prod(v.get_shape().as_list()))
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'vanilla')
    ]
    mobile_params = [
        (v.name, np.prod(v.get_shape().as_list()))
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'mobile')
    ]

    print("VANILLA CONV BLOCK")
    total_vanilla_params = sum([p[1] for p in vanilla_params])
    for p in vanilla_params:
        print("Variable {0}: number of params = {1}".format(p[0], p[1]))
    print("Total number of params =", total_vanilla_params)
    print()

    print("MOBILENET CONV BLOCK")
    total_mobile_params = sum([p[1] for p in mobile_params])
    for p in mobile_params:
        print("Variable {0}: number of params = {1}".format(p[0], p[1]))
    print("Total number of params =", total_mobile_params)
    print()

    print("{0:.3f}x parameter reduction".format(total_vanilla_params /
                                             total_mobile_params))


# Your solution should show the majority of the parameters in *MobileNet* block stem from the pointwise convolution.

# ## *MobileNet* SSD
# 
# In this section you'll use a pretrained *MobileNet* [SSD](https://arxiv.org/abs/1512.02325) model to perform object detection. You can download the *MobileNet* SSD and other models from the [TensorFlow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) (*note*: we'll provide links to specific models further below). [Paper](https://arxiv.org/abs/1611.10012) describing comparing several object detection models.
# 
# Alright, let's get into SSD!

# ### Single Shot Detection (SSD)
# 
# Many previous works in object detection involve more than one training phase. For example, the [Faster-RCNN](https://arxiv.org/abs/1506.01497) architecture first trains a Region Proposal Network (RPN) which decides which regions of the image are worth drawing a box around. RPN is then merged with a pretrained model for classification (classifies the regions). The image below is an RPN:
# 
# ![Faster-RCNN Visual](./assets/faster-rcnn.png)

# The SSD architecture is a single convolutional network which learns to predict bounding box locations and classify the locations in one pass. Put differently, SSD can be trained end to end while Faster-RCNN cannot. The SSD architecture consists of a base network followed by several convolutional layers: 
# 
# ![SSD Visual](./assets/ssd_architecture.png)
# 
# **NOTE:** In this lab the base network is a MobileNet (instead of VGG16.)
# 
# #### Detecting Boxes
# 
# SSD operates on feature maps to predict bounding box locations. Recall a feature map is of size $D_f * D_f * M$. For each feature map location $k$ bounding boxes are predicted. Each bounding box carries with it the following information:
# 
# * 4 corner bounding box **offset** locations $(cx, cy, w, h)$
# * $C$ class probabilities $(c_1, c_2, ..., c_p)$
# 
# SSD **does not** predict the shape of the box, rather just where the box is. The $k$ bounding boxes each have a predetermined shape. This is illustrated in the figure below:
# 
# ![](./assets/ssd_feature_maps.png)
# 
# The shapes are set prior to actual training. For example, In figure (c) in the above picture there are 4 boxes, meaning $k$ = 4.

# ### Exercise 2 - SSD Feature Maps
# 
# It would be a good exercise to read the SSD paper prior to a answering the following questions.
# 
# ***Q: Why does SSD use several differently sized feature maps to predict detections?***

# A: Your answer here
# 
# **[Sample answer](./exercise-solutions/e2.md)**

# The current approach leaves us with thousands of bounding box candidates, clearly the vast majority of them are nonsensical.
# 
# ### Exercise 3 - Filtering Bounding Boxes
# 
# ***Q: What are some ways which we can filter nonsensical bounding boxes?***

# A: Your answer here
# 
# **[Sample answer](./exercise-solutions/e3.md)**

# #### Loss
# 
# With the final set of matched boxes we can compute the loss:
# 
# $$
# L = \frac {1} {N} * ( L_{class} + L_{box})
# $$
# 
# where $N$ is the total number of matched boxes, $L_{class}$ is a softmax loss for classification, and $L_{box}$ is a L1 smooth loss representing the error of the matched boxes with the ground truth boxes. L1 smooth loss is a modification of L1 loss which is more robust to outliers. In the event $N$ is 0 the loss is set 0.
# 
# 

# ###  SSD Summary
# 
# * Starts from a base model pretrained on ImageNet. 
# * The base model is extended by several convolutional layers.
# * Each feature map is used to predict bounding boxes. Diversity in feature map size allows object detection at different resolutions.
# * Boxes are filtered by IoU metrics and hard negative mining.
# * Loss is a combination of classification (softmax) and dectection (smooth L1)
# * Model can be trained end to end.

# ## Object Detection Inference
# 
# In this part of the lab you'll detect objects using pretrained object detection models. You can download the latest pretrained models from the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), although do note that you may need a newer version of TensorFlow (such as v1.8) in order to use the newest models.
# 
# We are providing the download links for the below noted files to ensure compatibility between the included environment file and the models.
# 
# [SSD_Mobilenet 11.6.17 version](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz)
# 
# [RFCN_ResNet101 11.6.17 version](http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_11_06_2017.tar.gz)
# 
# [Faster_RCNN_Inception_ResNet 11.6.17 version](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz)
# 
# Make sure to extract these files prior to continuing!

# In[ ]:


# Frozen inference graph files. NOTE: change the path to where you saved the models.
SSD_GRAPH_FILE = 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
RFCN_GRAPH_FILE = 'rfcn_resnet101_coco_11_06_2017/frozen_inference_graph.pb'
FASTER_RCNN_GRAPH_FILE = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/frozen_inference_graph.pb'


# Below are utility functions. The main purpose of these is to draw the bounding boxes back onto the original image.

# In[ ]:


# Colors (one for each class)
cmap = ImageColor.colormap
print("Number of colors =", len(cmap))
COLOR_LIST = sorted([c for c in cmap.keys()])

#
# Utility funcs
#

def filter_boxes(min_score, boxes, scores, classes):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)
    
    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes

def to_image_coords(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].
    
    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width
    
    return box_coords

def draw_boxes(image, boxes, classes, thickness=4):
    """Draw bounding boxes on the image"""
    draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        class_id = int(classes[i])
        color = COLOR_LIST[class_id]
        draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)
        
def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


# Below we load the graph and extract the relevant tensors using [`get_tensor_by_name`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name). These tensors reflect the input and outputs of the graph, or least the ones we care about for detecting objects.

# In[ ]:


detection_graph = load_graph(SSD_GRAPH_FILE)
# detection_graph = load_graph(RFCN_GRAPH_FILE)
# detection_graph = load_graph(FASTER_RCNN_GRAPH_FILE)

# The input placeholder for the image.
# `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Each box represents a part of the image where a particular object was detected.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

# The classification of the object (integer id).
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')


# Run detection and classification on a sample image.

# In[ ]:


# Load a sample image.
image = Image.open('./assets/sample1.jpg')
image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

with tf.Session(graph=detection_graph) as sess:                
    # Actual detection.
    (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes], 
                                        feed_dict={image_tensor: image_np})

    # Remove unnecessary dimensions
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)

    confidence_cutoff = 0.8
    # Filter boxes with a confidence score less than `confidence_cutoff`
    boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

    # The current box coordinates are normalized to a range between 0 and 1.
    # This converts the coordinates actual location on the image.
    width, height = image.size
    box_coords = to_image_coords(boxes, height, width)

    # Each class with be represented by a differently colored box
    draw_boxes(image, box_coords, classes)

    plt.figure(figsize=(12, 8))
    plt.imshow(image) 


# ## Timing Detection
# 
# The model zoo comes with a variety of models, each its benefits and costs. Below you'll time some of these models. The general tradeoff being sacrificing model accuracy for seconds per frame (SPF).

# In[ ]:


def time_detection(sess, img_height, img_width, runs=10):
    image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
    detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')

    # warmup
    gen_image = np.uint8(np.random.randn(1, img_height, img_width, 3))
    sess.run([detection_boxes, detection_scores, detection_classes], feed_dict={image_tensor: gen_image})
    
    times = np.zeros(runs)
    for i in range(runs):
        t0 = time.time()
        sess.run([detection_boxes, detection_scores, detection_classes], feed_dict={image_tensor: image_np})
        t1 = time.time()
        times[i] = (t1 - t0) * 1000
    return times


# In[ ]:


with tf.Session(graph=detection_graph) as sess:
    times = time_detection(sess, 600, 1000, runs=10)


# In[ ]:


# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)
plt.title("Object Detection Timings")
plt.ylabel("Time (ms)")

# Create the boxplot
plt.style.use('fivethirtyeight')
bp = ax.boxplot(times)


# ### Exercise 4 - Model Tradeoffs
# 
# Download a few models from the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and compare the timings.

# ## Detection on a Video
# 
# Finally run your pipeline on [this short video](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/advanced_deep_learning/driving.mp4).

# In[ ]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[ ]:


HTML("""
<video width="960" height="600" controls>
  <source src="{0}" type="video/mp4">
</video>
""".format('driving.mp4'))


# ### Exercise 5 - Object Detection on a Video
# 
# Run an object detection pipeline on the above clip.

# In[ ]:


clip = VideoFileClip('driving.mp4')


# In[ ]:


# TODO: Complete this function.
# The input is an NumPy array.
# The output should also be a NumPy array.
def pipeline(img):
    pass


# **[Sample solution](./exercise-solutions/e5.py)**

# In[ ]:


with tf.Session(graph=detection_graph) as sess:
    image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
    detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')
    
    new_clip = clip.fl_image(pipeline)
    
    # write to file
    new_clip.write_videofile('result.mp4')


# In[ ]:


HTML("""
<video width="960" height="600" controls>
  <source src="{0}" type="video/mp4">
</video>
""".format('result.mp4'))


# ## Further Exploration
# 
# Some ideas to take things further:
# 
# * Finetune the model on a new dataset more relevant to autonomous vehicles. Instead of loading the frozen inference graph you'll load the checkpoint.
# * Optimize the model and get the FPS as low as possible.
# * Build your own detector. There are several base model pretrained on ImageNet you can choose from. [Keras](https://keras.io/applications/) is probably the quickest way to get setup in this regard.
# 

# In[ ]:




