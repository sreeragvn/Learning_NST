
# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tf.keras.backend.set_floatx('float64')

import os
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

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


content_image =load_img("sub_content2.jpg", 1024)
style_image =load_img("sub_style2.jpg", 1024)

# %%
content_image_shape = tf.shape(content_image)
style_image_shape = tf.shape(style_image)

# #!/usr/bin/python
# import sys
# import shutil

# # Get directory name
# mydir = os.path.dirname("/home/fpds01/.keras/models/vgg19_weights_tf_dim_ordering_tf_kernels.h5")


    
# %%
# YOUR CODE
# Try to remove the tree; if it fails, throw an error using try...except.
# try:
#     shutil.rmtree(mydir)
# except OSError as e:
#     print("Error: %s - %s." % (e.filename, e.strerror))
vgg = tf.keras.applications.VGG19(include_top=False, weights = "imagenet")
for layer in vgg.layers:
    print(layer.name)

# %%
# YOUR CODE
content_layers = []
style_layers = []
for layer in vgg.layers:
  if (layer.name == "block4_conv2"):
    content_layers.append(layer.name)
  elif (layer.name.endswith("conv1")):
    style_layers.append(layer.name)

# %%
# YOUR CODE
def vgg_layers(layer_names):

    # # Try to remove the tree; if it fails, throw an error using try...except.
    # try:
    #     shutil.rmtree(mydir)
    # except OSError as e:
    #     print("Error: %s - %s." % (e.filename, e.strerror))  
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    
    outputs = []
    for layer in layer_names:
      outputs.append(vgg.get_layer(layer).output)

    model = tf.keras.Model([vgg.input], outputs)
    return model

# %%
vgg_layers(style_layers).output

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


extractor = StyleContentModel(style_layers, content_layers)


style_targets = extractor(style_image)['style']

# %%
style_targets["block1_conv1"]

content_targets = extractor(content_image)['content']


# %%
content_targets["block4_conv2"][0][0]


# %%
# YOUR CODE
image =  tf.Variable(content_image)

print(image)

# %%
# YOUR CODE
content_weight = 1e-3
style_weight = 1

# %%
# YOUR CODE
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# %%
# YOUR CODE
def clip_0_1(image):
  return tf.clip_by_value( image, 0 , 1)


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


# %%
# YOUR CODE
tv_weight = 1e3

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

image =  tf.Variable(content_image)

loops = 100
iters_per_loop = 10

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
