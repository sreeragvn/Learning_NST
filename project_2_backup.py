# %% [markdown]
# # Project 2: Neural Style Transfer
# ##### 
# ##### Submission due on December 11th 
# ##### 
# 
# <table style="width:100%">
#     <tr>
#         <th><img src="altgebaeude.jpg?666" alt="" style="width: 333px;"></th>
#         <th><img src="vangogh.jpg?999" alt="" style="width: 333px;"></th>
#         <th><img src="altgebaeude_stilisiert.png?555" alt="" style="width: 333px;"></th>
#     </tr>
#     <tr>
#         <th>Content Image $\mathbf{c}$</th>
#         <th>Style Image $\mathbf{s}$</th>
#         <th>Generated Image $\mathbf{x}$</th>
#     </tr>
# </table> 
# 
# #### 
# 
# Welcome to *Project 2: Neural Style Transfer*! Neural style transfer is an approach to modify a given image (the *Content Image*, usually a photo) such that it assimilates another given image (the *Style Image*, usually a painting) in terms of style. In doing that, of course we need an explicit definition of what we mean by *style*: A specific loss function is minimized which incorporates a pretrained convolutional neural network (CNN). A subset of layers of the CNN are used to compute a distance between the sought image (the *Generated Image*) and the *Content Image*, and between the *Generated Image* and the *Style Imgage*, respectively.
# 
# The underlying idea originates in the paper <a href="https://arxiv.org/pdf/1508.06576.pdf">A Neural Algorithm of Artistic Style</a> by Leon Gatys, Alexander Ecker and Matthias Bethge. It is assumed that outputs of CNN layers indicate whether or not areas of an input image contain certain learned patterns. Each filter in a CNN layer represents one pattern, and the deeper a filter is located in the network, the more complex and abstract can be the extracted information. While layer outputs (activations) can be used directly to derive a distance between content image and generated image, something more sophisticated is needed to measure similarity in style. To establish a style based distance, correlations (inner products) between all output channels of a CNN layer are computed and stored in a *Gram Matrix*. A comparison between *Style Image* and *Generated Image* is then accomplished in terms of several Gram matrices relating to different layers. More details can be found below, in the above-mentioned paper, and in the Stud.IP Courseware.
# 
# In this project, you will use the readily trained and available CNN <a href="https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg19">VGG19</a> (see also <a href="https://arxiv.org/pdf/1409.1556.pdf">Very Deep Convolutional Networks for Large-Scale Image Recognition</a> by Karen Simonyan Andrew Zisserman). In other words, our goal here is **not** to learn parameters of a neural network. We solely seek the *Generated Image*. However, you will learn fundamentals for an advanced usage of Keras and TensorFlow. To compute CNN activations and Gram matrices from inputs, you will need the <a href="https://studip.tu-braunschweig.de/plugins.php/courseware/courseware?cid=20c4897dea349b14eca5d904fda9504d&selected=1400901">Keras Functional API</a> as well as the <a href="https://studip.tu-braunschweig.de/plugins.php/courseware/courseware?cid=20c4897dea349b14eca5d904fda9504d&selected=1400902">Keras Subclassing API</a>. These two enable you to create custom models in Keras and surpass the possibilities of the <a href="https://studip.tu-braunschweig.de/plugins.php/courseware/courseware?cid=20c4897dea349b14eca5d904fda9504d&selected=1400900">Keras Sequential API</a> by far. Moreover, you will use <a href="https://studip.tu-braunschweig.de/plugins.php/courseware/courseware?cid=20c4897dea349b14eca5d904fda9504d&selected=1400905">TensorFlow functions</a> and <a href="https://studip.tu-braunschweig.de/plugins.php/courseware/courseware?cid=20c4897dea349b14eca5d904fda9504d&selected=1400896">automatic differentiation</a> in order to implement <a href="https://studip.tu-braunschweig.de/plugins.php/courseware/courseware?cid=20c4897dea349b14eca5d904fda9504d&selected=1400904">custom training loops</a> in TensorFlow.

# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tf.keras.backend.set_floatx('float64')

import os
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

#!/usr/bin/python
import sys
import shutil

# Get directory name
mydir = input("/home/fpds01/.keras/models/")


# %% [markdown]
# ## 2.1 Loading and formatting images with TensorFlow

# %% [markdown]
# 1. Write a function ```load_img(path_to_img, max_dim)``` that reads and resizes an image. The function shall return a 4D-tensor ```img``` of type ```tf.float64``` with values in the range ```[0,1]```. Height and width of the image shall be scaled such that the maximum of ```img.shape[1]``` (height in pixels) and ```img.shape[2]``` (width in pixels) is exactly ```max_dim```. Moreover, the output should feature ```img.shape[0] = 1``` (batch size) and ```img.shape[3] = 3``` (number of channels). Proceed as follows:
# 
#     1.1 Use ```tf.io.read_file```, ```tf.image.decode_image``` and ```tf.image.convert_image_dtype``` (in this order), to read and convert the input image.
#     
#     1.2 Use ```tf.image.resize``` to scale height and width by the same factor. Make sure that the maximum 
#     
#     1.3 Use ```tf.expand_dims``` to add the batch dimension.
#     
#     1.4 Use ```tf.cast``` to make sure that the output has type ```tf.float64```.
# 

# %%
# YOUR CODE
def load_img(path_to_img, max_dim):
  
  img = tf.io.read_file(path_to_img)
  
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, dtype = tf.float64)
  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  # print(shape)
  long_dim = max(shape)
  # print(long_dim)
  scale = max_dim / long_dim
  new_shape = tf.cast(shape * scale, tf.int32)
  # img = tf.image.resize(img, new_shape, preserve_aspect_ratio=True)
  img = tf.image.resize(img, new_shape)
  img = tf.expand_dims(img, axis=0)
  return tf.cast(img, tf.float64)


# %% [markdown]
# 2. Call ```load_img``` with ```max_dim = 512``` to load ```altgebaeude.jpg``` and ```vangogh.jpg``` and store them in variables ```content_image``` and ```style_image```, respectively.
# 
# 
# 3. Use ```plt.imshow``` to display both images.

# %%
# YOUR CODE
content_image =load_img("altgebaeude.jpg", 512)
style_image =load_img("vangogh.jpg", 512)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(content_image[0])
plt.title('Content Image')

plt.subplot(1, 2, 2)
plt.imshow(style_image[0])
plt.title('Style Image')
plt.tight_layout()
plt.show()

# %% [markdown]
# 4. Apply ```tf.shape``` to ```content_image``` and ```style_image``` and diplay both outputs.

# %%
content_image_shape = tf.shape(content_image)
style_image_shape = tf.shape(style_image)
print(content_image_shape)
print(style_image_shape)

# %% [markdown]
# **Checkpoint:** Your output should look like this:
# 
# ```
# tf.Tensor([  1 337 512   3], shape=(4,), dtype=int32)
# tf.Tensor([  1 405 512   3], shape=(4,), dtype=int32)
# ```

# %% [markdown]
# ## 2.2 Using pretrained models in Keras
# 
# ```tf.keras.applications``` lets you load various readily trained neural networks, for example ```VGG19``` which we are going to use here. Before starting with Neural Style Transfer, let us see how to use a pretrained model for multiclass classification.
# 
# 5. Call ```tf.keras.applications.VGG19``` with arguments ```include_top = True``` and ```weights = "imagenet"```, and store the output in a variable ```vgg```.
# 
# The first-mentioned argument is to get both the preceding convolutional layers *and* the subsequent fully connected layers at the top of ```VGG19```. Vice versa, ```include_top = False``` lets you load the network without fully connected layers. The version without fully connected layers has the advantage that images of arbitrary height and width are accepted as input and can be propagated through the network (we will make use of this for Neural Style Transfer). The variant you should have loaded now accepts only ```224 x 224``` images. The latter argument ```weights = "imagenet"``` implies that a network is loaded that was pretrained on the ```imagenet``` dataset. By means of ```weights = None```, you can also get a randomly initialized version.

# %%

# Try to remove the tree; if it fails, throw an error using try...except.
try:
    shutil.rmtree(mydir)
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))
vgg = tf.keras.applications.VGG19(include_top=True, weights = "imagenet")

# %% [markdown]
# 6. Use ```tf.keras.models.Model.summary``` to get an overview of the model.

# %%
# YOUR CODE
vgg.summary()

# %% [markdown]
# Each model in ```keras.applications``` has an own preprocessing function that should be applied to inputs before feeding them into the network. In case of ```VGG19```, the preprocessing function expects an image with integer pixel values between 0 and 255 as input.
# 
# 7. Apply ```tf.keras.applications.vgg19.preprocess_input``` to a version of ```content_image``` with accordingly scaled pixel values and store the output in ```content_image_pp```. Afterwards, use ```tf.image.resize``` to rescale ```content_image_pp``` such that you can apply ```vgg``` to the result ```content_image_rs```. Do the latter and store the output of ```vgg``` in ```predicted_probabilities```.

# %%
# YOUR CODE
pixelvalue = content_image*255
content_image_pp = tf.keras.applications.vgg19.preprocess_input(pixelvalue)
content_image_rs = tf.image.resize(content_image_pp, size = [224, 224], preserve_aspect_ratio=False)
predicted_probabilities = vgg.predict(content_image_rs)

# %% [markdown]
# 8. Now call ```tf.keras.applications.vgg19.decode_predictions``` with arguments ```predicted_probabilities.numpy()``` and ```top = 10``` to display the top 10 classes and associated probabilities for the given input. Store the output in ```predicted_top_10``` and display ```[(class_name, prob) for (number, class_name, prob) in predicted_top_10[0]]```.

# %%
# YOUR CODE
predicted_top_10 = tf.keras.applications.vgg19.decode_predictions(predicted_probabilities, top = 10)
[(class_name, prob) for (number, class_name, prob) in predicted_top_10[0]]

# %% [markdown]
# **Checkpoint:** You should have got the following output:
# ```
# [('palace', 0.9914493749464944),
#  ('triumphal_arch', 0.00267970467587212),
#  ('monastery', 0.0013206769341534597),
#  ('library', 0.0011177809930798567),
#  ('fountain', 0.0008800198818778909),
#  ('church', 0.0004445435231895076),
#  ('castle', 0.0003099415218987508),
#  ('obelisk', 0.00029764423272215536),
#  ('dome', 0.00022929634370079404),
#  ('cinema', 0.00022750031564633795)]
#  ```
# 
# So, according to ```VGG19```, our old building is most probably a palace!
# 
# 9. Repeat 7. und 8. with ```style_image``` as input.

# %%
# YOUR CODE
# YOUR CODE
pixelvalue_style = style_image*255
style_image_pp = tf.keras.applications.vgg19.preprocess_input(pixelvalue_style)
style_image_rs = tf.image.resize(style_image_pp, size = [224, 224], preserve_aspect_ratio=False)
predicted_probabilities = vgg.predict(style_image_rs)

predicted_top_10 = tf.keras.applications.vgg19.decode_predictions(predicted_probabilities, top = 10)
[(class_name, prob) for (number, class_name, prob) in predicted_top_10[0]]

# %% [markdown]
# ## 2.3 Defining a feature extractor
# 
# Now, we come to the main part of this project: *Neural Style Transfer*! To evaluate the objective
# 
# $$J(\mathbf{x}, \mathbf{c}, \mathbf{s}) = \alpha \ J_\mathrm{content}(\mathbf{x}, \mathbf{c}) + \beta \ J_\mathrm{style}(\mathbf{x}, \mathbf{s})\qquad (*)$$
# 
# we need a *feature extractor*. The feature extractor shall provide us outputs $\mathrm{a}^{[\ell](\mathbf{x})}$ and $\mathrm{a}^{[\ell](\mathbf{c})}$ of selected convolutional layers of ```VGG19``` that we need to evaluate the content loss:
# 
# $$J_\mathrm{content}(\mathbf{x}, \mathbf{c}) = \frac{1}{2 \vert L_\mathrm{content}\vert} \sum_{\ell\in L_\mathrm{content}} \Vert \mathrm{a}^{[\ell](\mathbf{x})} - \mathrm{a}^{[\ell](\mathbf{c})} \Vert_2^2\qquad (**)$$
# 
# The set $L_\mathrm{content}$ contains all indices of layers that shall be considered in the content loss. Accordingly, $L_\mathrm{style}$ refers to layers that shall be considered in the style loss.
# 
# Respective outputs are used to compute gram matrices $\mathrm{G}^{[\ell](\mathbf{x})}$ and $\mathrm{G}^{[\ell](\mathbf{s})}$ that we need to evaluate the style loss:
# 
# $$J_\mathrm{style}(\mathbf{x}, \mathbf{s}) = \frac{1}{4 \vert L_\mathrm{style}\vert} \sum_{\ell\in L_\mathrm{style}} \Vert \mathrm{G}^{[\ell](\mathbf{x})} - \mathrm{G}^{[\ell](\mathbf{s})} \Vert_2^2\qquad (***)$$
# 
# Therein, the gram matrices are computed via
# 
# $$\mathrm{G}^{[\ell](\mathbf{x})}_{i, j} = \frac{1}{n^{[\ell]}_H n^{[\ell]}_W n^{[\ell]}_C} \langle\mathrm{a}^{[\ell](\mathbf{x})}_{i}, \mathrm{a}^{[\ell](\mathbf{x})}_{j}\rangle\quad\text{für}\quad i, j =1,\dots,n^{[\ell]}_C\qquad (****)$$
# 
# and $\mathrm{a}^{[\ell](\mathbf{x})}_{i}$ and $\mathrm{a}^{[\ell](\mathbf{x})}_{j}$ represent the $i$-th and $j$-th channel of $\mathrm{a}^{[\ell](\mathbf{x})}$, respectively.
# 
# 
# 10. Load ```VGG19``` again, this time with ```include_top = False```. Afterwards, display the names of all layers in ```vgg.layers```.

# %%
# YOUR CODE
# Try to remove the tree; if it fails, throw an error using try...except.
try:
    shutil.rmtree(mydir)
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))
vgg = tf.keras.applications.VGG19(include_top=False, weights = "imagenet")
for layer in vgg.layers:
    print(layer.name)

# %% [markdown]
# **Checkpoint:** Your output should look as follows:
# ```
# input_2
# block1_conv1
# block1_conv2
# block1_pool
# block2_conv1
# block2_conv2
# block2_pool
# block3_conv1
# block3_conv2
# block3_conv3
# block3_conv4
# block3_pool
# block4_conv1
# block4_conv2
# block4_conv3
# block4_conv4
# block4_pool
# block5_conv1
# block5_conv2
# block5_conv3
# block5_conv4
# block5_pool
# ```
# 
# This output includes all layers, that can be added to $L_\mathrm{content}$ and $L_\mathrm{style}$. For the moment, let us fix the following:
# 
# 11. Create two lists ```content_layers``` and ```style_layers```. Initialize ```content_layers``` with a single element ```"block4_conv2"``` and ```style_layers``` with five elements ```"block1_conv1", ..., "block5_conv1"```.

# %%
# YOUR CODE
content_layers = []
style_layers = []
for layer in vgg.layers:
  if (layer.name == "block4_conv2"):
    content_layers.append(layer.name)
  elif (layer.name.endswith("conv1")):
    style_layers.append(layer.name)

# content_layers = ["block4_conv2"]
# style_layers = ["block1_conv1","block2_conv1","block3_conv1","block4_conv1","block5_conv1"]

# for i in style_layers:
#   print(i.name)


# %% [markdown]
# 12. Write a function ```vgg_layers``` that takes as input a list of strings ```layer_names``` and returns an object ```model``` of type ```keras.models.Model```. Start by loading a pretrained ```VGG19``` without fully connected layers. Then, use the Keras Functional API to create a model ```model``` that has the same input layer as ```VGG19``` and features a list of output layers corresponding to the subset of ```VGG19``` layers specified in ```layer_names```.

# %%
# YOUR CODE
def vgg_layers(layer_names):

    # Try to remove the tree; if it fails, throw an error using try...except.
    try:
        shutil.rmtree(mydir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))  
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    
    outputs = []
    for layer in layer_names:
      outputs.append(vgg.get_layer(layer).output)

    model = tf.keras.Model([vgg.input], outputs)
    return model

# %%
vgg_layers(style_layers).output

# %% [markdown]
# **Checkpoint:** The above call of ```vgg_layers(style_layers).output``` should create the following output:
# ```
# [<KerasTensor: shape=(None, None, None, 64) dtype=float64 (created by layer 'block1_conv1')>,
#  <KerasTensor: shape=(None, None, None, 128) dtype=float64 (created by layer 'block2_conv1')>,
#  <KerasTensor: shape=(None, None, None, 256) dtype=float64 (created by layer 'block3_conv1')>,
#  <KerasTensor: shape=(None, None, None, 512) dtype=float64 (created by layer 'block4_conv1')>,
#  <KerasTensor: shape=(None, None, None, 512) dtype=float64 (created by layer 'block5_conv1')>]
#  ```

# %% [markdown]
# 13. Write a function ```gram_matrix``` that gets as input a 4D-tensor ```input_tensor``` (corresponding to $\mathbf{a}^{[\ell](\mathbf{x})}$ and $\mathbf{a}^{[\ell](\mathbf{s})}$, respectively) and returns the Gram matrix of this tensor according to $(****)$. To that end, apply the function ```tf.linalg.einsum``` conveniently as a first step. Keep in mind that the leading dimension of ```input_tensor``` represents the batch size and should remain untouched during Gram matrix computation.

# %%
def gram_matrix(input_tensor):
  # print( tf.shape(input_tensor))
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  # print(input_shape)
  num_locations = tf.cast(input_shape[1]*input_shape[2]*input_shape[3], tf.float64)
  return result/num_locations

# %%
gram_matrix(vgg_layers(style_layers)(content_image)[0])

# %% [markdown]
# **Checkpoint:** The above call of ```gram_matrix(vgg_layers(style_layers)(content_image)[0])``` should produce the following output:
# ```
# <tf.Tensor: shape=(1, 64, 64), dtype=float64, numpy=
# array([[[0.01065388, 0.00390675, 0.00425856, ..., 0.00097763,
#          0.00566108, 0.00761762],
#         [0.00390675, 0.0019675 , 0.00208546, ..., 0.00022037,
#          0.00267012, 0.00339072],
#         [0.00425856, 0.00208546, 0.00227649, ..., 0.00021288,
#          0.00297196, 0.00380253],
#         ...,
#         [0.00097763, 0.00022037, 0.00021288, ..., 0.00061053,
#          0.00064032, 0.0006466 ],
#         [0.00566108, 0.00267012, 0.00297196, ..., 0.00064032,
#          0.00492798, 0.00563125],
#         [0.00761762, 0.00339072, 0.00380253, ..., 0.0006466 ,
#          0.00563125, 0.00693724]]])>
# ```

# %% [markdown]
# 14. Complete the method ```call``` of the class ```StyleContentModel```. Proceed as follows:
# 
#     14.1 The argument ```inputs``` is again a 4D-tensor with values between zero and one. Apply ```tf.keras.applications.vgg19.preprocess_input``` to an appropriately scaled version of ```inputs``` (do not change height and width) and store the result in ```preprocessed_input```.
#     
#     14.2 Apply the attribute ```vgg``` to ```preprocessed_input``` and store the result in ```outputs```.
#     
#     14.3 Subdivide the list ```outputs``` into two lists ```style_outputs``` and ```content_outputs```. Before you do that, take a look at the initialization of ```vgg```.
#     
#     14.4 Replace all elements of ```style_outputs``` by associated Gram matrices.

# %%
# replace all occurences of None
class StyleContentModel(tf.keras.models.Model):  
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg =  vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
        
    def call(self, inputs):
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs*255)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}

# %% [markdown]
# **Remark:** You have finally used all three Keras APIs! In Project 1, you have built models using the *Keras Sequential API*. Above in 12. you used the *Keras Functional API* and in 14.4 the *Keras Subclassing API*. The flexibility of these APIs increases in the above order: The Sequential API is very easy to use, but only allows for models with a single input tensor, a single output tensor, and a number of sequential layers in between. The Functional API allows for multiple input and output tensors (we exploited the latter in 12.) and it is possible, for example, to build so-called skip connections where activations skip one ore more layers. The possibilities of the Subclassing API go far beyond that. Arbitrary TensorFlow functions (in our case to compute the Gram matrix in 14.4) as well as control structures like ```for``` and ```while``` can be used inside the ```call``` method. A drawback of the Subclassing API is that created models cannot be saved and loaded without extra effort. Unlike in case of the other two APIs, we need to create an instance of our (sub)class in order to get a model by means of the Subclassing API.
# 
# 15. Create an instance ```extractor``` of ```StyleContentModel```. In doing that, submit the lists ```style_layers``` and ```content_layers``` from above to the constructor.

# %%
# YOUR CODE
extractor = StyleContentModel(style_layers, content_layers)

# %% [markdown]
# Now we are ready to implement the losses $(*)$, $(**)$ and $(***)$ with the help of ```extractor```!

# %% [markdown]
# ## 2.4 Minimizing the loss
# 
# In the course of Neural Style Transfer we actually try to solve the optimization problem
# 
# $$\min_{\mathbf{x}} J(\mathbf{x}, \mathbf{c}, \mathbf{s})$$
# 
# where $\mathbf{c}$ and $\mathbf{s}$ represent *Content Image* and *Style Image*, respectively. As these two images are given, all $\mathbf{a}^{[\ell](\mathbf{c})}$ in $(**)$ and all $\mathbf{G}^{[\ell](\mathbf{s})}$ in $(***)$ are constants that we will compute now.
# 
# 16. Apply ```extractor``` to ```style_image``` to extract associated Gram matrices and store the result in ```style_targets```. The result must be a list with items $\mathbf{G}^{[\ell](\mathbf{s})}$ for multiple values of $\ell$.

# %%
# YOUR CODE
style_targets = extractor(style_image)['style']

# %%
style_targets["block1_conv1"]

# %% [markdown]
# **Checkpoint:** The above call of ```style_targets["block1_conv1"]``` should produce the following output:
# ```
# <tf.Tensor: shape=(1, 64, 64), dtype=float64, numpy=
# array([[[ 45.92743055,  20.06590504,   9.71989013, ...,  30.17798489,
#           10.98088872,  15.10200837],
#         [ 20.06590504,  45.85620007,  13.65396233, ...,  21.29668717,
#           14.39884349,  10.82709402],
#         [  9.71989013,  13.65396233,   8.18387814, ...,   2.56935254,
#            8.03212506,   8.51876579],
#         ...,
#         [ 30.17798489,  21.29668717,   2.56935254, ..., 198.46450743,
#           20.21175996,   9.80212628],
#         [ 10.98088872,  14.39884349,   8.03212506, ...,  20.21175996,
#           21.53061025,  15.80490754],
#         [ 15.10200837,  10.82709402,   8.51876579, ...,   9.80212628,
#           15.80490754,  16.56110532]]])>
# ```

# %% [markdown]
# 17. Apply ```extractor``` to ```content_image``` to extract associated activations and store the result in ```content_targets```. The result must be a list with items $\mathbf{a}^{[\ell](\mathbf{c})}$ for different values of $\ell$.

# %%
# YOUR CODE
content_targets = extractor(content_image)['content']


# %%
content_targets["block4_conv2"][0][0]

# %% [markdown]
# **Checkpoint:** The above call of ```content_targets["block1_conv1"][0][0]``` should produce the following output:
# ```
# <tf.Tensor: shape=(64, 512), dtype=float64, numpy=
# array([[725.71265434,   0.        ,  91.28209556, ...,  43.47577693,
#         375.02083088,   0.        ],
#        [  0.        ,   0.        , 353.31359666, ...,   0.        ,
#         351.77779676,   0.        ],
#        [  0.        ,   0.        , 490.15692419, ...,   0.        ,
#         460.51096857,   0.        ],
#        ...,
#        [  0.        ,   0.        , 178.48063073, ..., 161.33845743,
#         238.55003094,   0.        ],
#        [  0.        ,   0.        , 682.45457904, ..., 182.48885433,
#         390.16694736,   0.        ],
#        [  0.        ,   0.        , 607.08138405, ..., 595.88203066,
#         626.46038097,   0.        ]])>
# ```

# %% [markdown]
# 18. Initialize ```image``` (corresponding to $\mathbf{x}$) as a ```tf.Variable``` with value ```content_image```.

# %%
# YOUR CODE
image =  tf.Variable(content_image)

print(image)

# %% [markdown]
# 19. Initialize ```content_weight = 1e-3``` (corresponds to $\alpha$) and ```style_weight = 1``` (corresponds to $\beta$).

# %%
# YOUR CODE
content_weight = 1e-3
style_weight = 1

# %% [markdown]
# 20. Assign the optimizer ```tf.optimizers.Adam``` with parameters ```learning_rate=0.02```, ```beta_1=0.99``` and ```epsilon=1e-1``` to ```opt```.

# %%
# YOUR CODE
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# %% [markdown]
# 21. Write a function ```clip_0_1``` that takes a 4D-tensor ```image``` as input and clips the values of that tensor to the interval ```[0, 1]```. Use the function ```tf.clip_by_value```.

# %%
# YOUR CODE
def clip_0_1(image):
  return tf.clip_by_value( image, 0 , 1)


# %% [markdown]
# 22. Complete the function ```style_content_loss``` that corresponds to $(*)$ and will be called by ```train_step``` in the next step.

# %%
# replace all occurences of None
def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_sum((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()])
    style_loss *= (1/(4*len(style_layers)))
    content_loss = tf.add_n([tf.reduce_sum((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()])
    content_loss *= (1/(2*len(content_layers)))
    loss = style_weight * style_loss + content_weight * content_loss 
    return loss

# %% [markdown]
# 23. Complete the TensorFlow function ```train_step``` that computes a gradient of ```style_content_loss``` with respect to ```image``` and performs an associated gradient descent step.

# %%
# replace all occurences of None
@tf.function()
def train_step(image, opt):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight)
    grad = tape.gradient(loss, image)    
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
    return loss

# %% [markdown]
# **Remark:** The prefix ```@tf.function()``` creates a so-called TensorFlow function. Based on such functions, TensorFlow creates and optimizes computation graphs automatically at runtime which will often induce a drastic speed-up (through parallelization, etc.).

# %% [markdown]
# 24. Complement the following code with a call of ```train_step``` and start training. Watch the *Getting started with TensorBoard* video in the Stud.IP courseware to learn how to access the results.

# %%
loops = 100
iters_per_loop = 10
file_writer = tf.summary.create_file_writer('logs' + f'/stw{style_weight}_cow{content_weight}')
file_writer.set_as_default()

for loop in range(loops):
    tf.summary.image('image', data=image, step=loop * iters_per_loop)
    for it in range(iters_per_loop):
        # YOUR CODE
        loss = train_step(image,opt)
        tf.summary.scalar('loss', data=loss, step=loop * iters_per_loop + it)

# %%
import PIL.Image

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

file_name = 'stylized-image1.png'
tensor_to_image(image).save(file_name)

# %% [markdown]
# ## 2.5 Adding TV regularization

# %% [markdown]
# Images generated with *Neural Style Transfer* often feature high frequency parts which some viewers might perceive as unpleasant or noisy. One way to mitigate this effect is adding *TV regularization* to the overall loss ```(*)```. In a nutshell, this means that large color gradients of the generated image are penalized a little bit. The resulting image usually has less local variation and appears smoother than before. Luckily, TensorFlow features a readily implemented function to compute the total variation of images and we are now going to add this function to ```train_step```.

# %% [markdown]
# 25. Initialize ```tv_weight = 1e3```.

# %%
# YOUR CODE
tv_weight = 1e3

# %% [markdown]
# 26. Redefine ```train_step```. The only difference above should be an additional loss term ```tv_weight * tf.cast(tf.image.total_variation(image)[0], tf.float64)```.

# %%
# YOUR CODE
def train_step(image, opt, tv_weight):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight)
        addn_loss = (tv_weight * tf.cast(tf.image.total_variation(image)[0], tf.float64)) #Wow
        loss = loss + addn_loss
    grad = tape.gradient(loss, image)    
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
    return loss

# %% [markdown]
# 27. Initialize ```image``` again.

# %%
# YOUR CODE
image =  tf.Variable(content_image)

# print(image)

# %% [markdown]
# 28. Repeat step 24. and view the results. Don't forget to re-initialize ```opt```.

# %%
# YOUR CODE

file_writer = tf.summary.create_file_writer('logs' + f'/stw{style_weight}_cow{content_weight}')
file_writer.set_as_default()

for loop in range(loops):
    tf.summary.image('image', data=image, step=loop * iters_per_loop)
    for it in range(iters_per_loop):
        # YOUR CODE
        loss = train_step(image,opt, tv_weight)
        tf.summary.scalar('loss', data=loss, step=loop * iters_per_loop + it)

# %%
import PIL.Image

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

file_name = 'stylized-image2.png'
tensor_to_image(image).save(file_name)

# %% [markdown]
# 29. Submit this notebook not later than December 11th.

# %% [markdown]
# 30. Apply Neural Style Transfer to content and style images of your choice. Three (not more and not less) self-created images per group need to be submitted and are also due on Decmeber 11th.
# 
# The submitted images are intended to participate in a *Neural Art Contest* that will take place towards the end of the semester in the form of a public online poll. The guiding theme of the contest is *Nature*. So your content images should feature something from this context. Apart from using content and style images of your choice, there are several things you could try:
# 
# - Use a larger value of ```max_dim``` to produce higher-resolution images.
# - Use constant or random initialization for ```image``` instead of ```content_image```.
# - Use lists ```content_layers``` and ```style_layers``` that are different from the standard setting.
# - Modify ```content_weight```, ```style_weight``` and ```tv_weight```.
# - Replace ```tf.optimizers.Adam``` by another algorithm.
# - Perform more iterations.
# - Anything else that comes to your mind.
# 
# Importantly, you need to use your own code to create the images. Moreover, your submission needs to be reproducible, i.e., you are asked to submit all content, style and generated images Python scripts that can be run to reproduce your submitted images.
# 
# We recommend to use the GPU cluster, especially when you increase the image resolution.


