---
layout: post
comments: true
title: Hierarchical Label Explainability
author: Kosta Gjorgjievski, Won June Lee, Adrian McIntosh
date: 2024-01-01
---


> Understanding how hierarchical labels affect saliency maps can unlock new pathways for model transparency and interpretability. This post explores the motivation, existing methods, and implementation of our project on hierarchical label explainability.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Deep learning models often operate as black boxes, making their decisions difficult to interpret. Saliency maps, which highlight areas of an image contributing most to a model's decision, are a key tool for visualization. By studying how saliency maps change across levels of abstraction in hierarchical labels, we aim to enhance interpretability and provide finer-grained insights into model behavior. Our focus is on the iNatLoc500 dataset, a challenging benchmark with 500 species organized in a label hierarchy.

## Motivation
Model transparency is crucial for trust and accountability in AI systems. Details, such as whether a model correctly distinguishes between closely related categories (e.g., Bulldog and Husky), can reveal its true understanding. Hierarchical labeling offers a structured approach to interpretability, breaking down decisions into different levels of granularity. This project seeks to address: 

**How hierarchical labels influence saliency maps.**  
**Whether these insights improve model transparency and user trust.**

## Exisitng Methods
### Saliency Maps
Saliency maps highlight the regions in an input image that contribute most to the model's predictions. Areas with high brightness correspond to influential regions. Saliency is computed using gradients of the model's output with respect to input pixels:  
**Key References:**  
[1] Simonyan et al. (2014): Introduced visualization techniques for convolutional networks.  
[2] Samek et al. (2015): Evaluated methods for visualizing learned features in deep networks.

### Techniques Used
**Vanilla Backpropagation:** Computes gradients of the output with respect to input pixels to generate the raw saliency map.  
**Guided Backpropagation:** Filters gradients to only show positive contributions, improving focus on important areas.  
**Integrated Gradients:** Accumulates gradients along a path from a baseline input (e.g., black image) to the actual input, providing a more robust attribution.  
**Grad-CAM (Gradient-weighted Class Activation Mapping):** Utilizes feature maps from intermediate layers to produce coarse, class-discriminative localization.

### label Granularity and Accuracy
A study by Cole et al. (2022) investigated how hierarchical labeling affects accuracy. Using ResNet50 and iNatLoc500, they demonstrated the potential for granularity to improve interpretability. However, their implementation lacked accessible code, highlighting a gap we aim to address.


## Our Proejct
### Implementation
We used the VGG16 architecture, trained on the iNatLoc500 dataset, to explore hierarchical labels. The steps included:  
[1] **Gradient Computation:**
Calculated the gradient of the model’s output with respect to each pixel in the input image to generate saliency maps.
```
class TrainConfig(object):
    """Training configuration"""
    dropout_keep_prob = 1.0
    model_name = 'vgg_16'  # choose model 
    model = staticmethod(globals()[model_name])
    config_name = 'no_hue'  # choose training run
```
[2] **Saliency Map Extraction:**
Visualized the extracted saliency maps to identify key features at different label levels.
```
def predict(imgs, config):
    """Load most recent checkpoint, make predictions, compute saliency maps"""
    g = tf.Graph()
    with g.as_default():
        imgs_ph = tf.placeholder(dtype=tf.uint8, shape=(None, 56, 56, 3))
        logits = config.model(imgs_ph, config)
        top_pred = tf.reduce_max(logits, axis=1)
        top_5 = tf.nn.top_k(logits, k=5, sorted=True)
        # can't calculate gradient to integer, get float32 version of image:
        float_img = g.get_tensor_by_name('Cast:0')
        # calc gradient of top predicted class to image:
        grads = tf.gradients(top_pred, float_img)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            path = 'checkpoints/' + config.model_name + '/' + config.config_name
            saver.restore(sess, tf.train.latest_checkpoint(path))
            feed_dict = {imgs_ph: imgs}
            top_5_np, grads_np = sess.run([top_5, grads], feed_dict=feed_dict)
      
    return top_5_np, grads_np
```

### Dataset: iNatLoc500
**Species:** 500
**Hierarchy:** Multi-level label structure
**Training Images:** 128k
**Test Images:** 25k

## Experiements
Our experiments focused on comparing saliency maps across abstraction levels. Key findings included:  
Fine-grained categories (e.g., Bulldog vs. Husky) highlighted specific facial features.  
Coarse-grained categories (e.g., Dog vs. Cat) emphasized broader body outlines.  

## Results and Observations
### Visual Insigts:  
Saliency maps at finer-grained levels captured intricate features like fur texture.  
Coarser levels generalized to larger regions of the image.
### Challenges:  
Saliency maps sometimes failed to distinguish between visually similar species.

## Example Visualization
![YOLO]({{ '/assets/images/team14/saliencymap.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. Example of how saliency map works with different objects* [1].  
image from https://debuggercafe.com/saliency-maps-in-convolutional-neural-networks/

## Conclusion and Future Work
This project demonstrated the potential of hierarchical labels to enhance interpretability through saliency maps. However, challenges remain in scaling this approach to larger datasets and improving its robustness for similar categories. Future steps include:  
Incorporating multi-grained descriptors as suggested by Wang et al. (2015).  
Testing additional architectures like ResNet50 for performance comparisons.


## image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

Please cite the image if it is taken from other people's work.


### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |



### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
```


### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.


## Reference

[1] Simonyan, K., Vedaldi, A., & Zisserman, A. (2014). Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps. arXiv preprint arXiv:1312.6034.  
[2] Samek, W., Binder, A., Montavon, G., Bach, S., & Müller, K.-R. (2015). Evaluating the Visualization of What a Deep Neural Network Has Learned. arXiv preprint arXiv:1509.06321.  
[3] Zhang, Q., Wu, Y. N., & Zhu, S.-C. (2018). Interpretable Convolutional Neural Networks. arXiv preprint arXiv:1710.00935.  
[4] Cole, E. et al. (2022). On Label Granularity and Object Localization. In: Avidan, S., Brostow, G., Cissé, M., Farinella, G.M., Hassner, T. (eds) Computer Vision – ECCV 2022. ECCV 2022. Lecture Notes in Computer Science, vol 13670. Springer, Cham. https://doi.org/10.1007/978-3-031-20080-9_35  
[5] Wang, Dequan & Shen, Zhiqiang & Shao, Jie & Zhang, Wei & Xue, Xiangyang & Zhang, Zheng. (2015). Multiple Granularity Descriptors for Fine-Grained Categorization. 2399-2406. 10.1109/ICCV.2015.276.

---
