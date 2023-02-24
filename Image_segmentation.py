# %%
# Import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, callbacks
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from tensorflow.keras.callbacks import TensorBoard
import datetime

# %%
# 1. Load the data
# 1.1. Prepare an empty list for the images and masks

train_images = []
train_masks = []
test_images = []
test_masks = []

train_path = r"C:\Users\user\Desktop\capstone project\Image Segmentation\data-science-bowl-2018-2\train"
test_path = r"C:\Users\user\Desktop\capstone project\Image Segmentation\data-science-bowl-2018-2\test"

# %%
# 1.2. Load the images and masks for the train
train_image_dir = os.path.join(train_path, "inputs")
for train_image in os.listdir(train_image_dir):
    img = cv2.imread(os.path.join(train_image_dir, train_image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    train_images.append(img)

train_mask_dir = os.path.join(train_path, "masks")
for train_mask in os.listdir(train_mask_dir):
    mask = cv2.imread(os.path.join(train_mask_dir, train_mask), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (128, 128))
    train_masks.append(mask)

# 1.3. Load the images and masks for the test
test_image_dir = os.path.join(test_path, "inputs")
for test_image in os.listdir(test_image_dir):
    img = cv2.imread(os.path.join(test_image_dir, test_image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    test_images.append(img)

test_mask_dir = os.path.join(test_path, "masks")
for test_mask in os.listdir(test_mask_dir):
    mask = cv2.imread(os.path.join(test_mask_dir, test_mask), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (128, 128))
    train_masks.append(mask)

# %%
# 1.4. Convert the lists into numpy array
train_images_np = np.array(train_images)
train_masks_np = np.array(train_masks)
test_images_np = np.array(test_images)
test_masks_np = np.array(test_masks)

# %%
# 1.5. Check some examples
plt.figure(figsize=(10, 10))
for i in range(1, 4):
    plt.subplot(1, 3, i)
    plt.imshow(train_images_np[i])
    plt.axis("off")

plt.show()
# %%
# 2. Data preprocessing
# 2.1. Expand the mask dimension
train_masks_np_exp = np.expand_dims(train_masks_np, axis=-1)
test_masks_np_exp = np.expand_dims(test_masks_np, axis=-1)
# Check the mask output
print(np.unique(train_masks[0]))

# %%
# 2.2. Convert the mask values into class labels
converted_train_masks = np.round(train_masks_np_exp / 255).astype(np.int64)
converted_test_masks = np.round(test_masks_np_exp / 255).astype(np.int64)
# Check the mask output
print(np.unique(converted_train_masks[0]))

# %%
# 2.3. Normalize image pixels value
converted_train_images = train_images_np / 255.0
train_sample = converted_train_images[0]

converted_test_images = test_images_np / 255.0
test_sample = converted_test_images[0]

# %%
# 2.4. Convert the numpy arrays into tensor
x_train = tf.data.Dataset.from_tensor_slices(converted_train_images)
x_test = tf.data.Dataset.from_tensor_slices(converted_test_images)
y_train = tf.data.Dataset.from_tensor_slices(converted_train_masks)
y_test = tf.data.Dataset.from_tensor_slices(converted_test_masks)

# %%
# 2.5. Combine the images and masks using zip
train_dataset = tf.data.Dataset.zip((x_train, y_train))
test_dataset = tf.data.Dataset.zip((x_test, y_test))

# %%
# 2.6. Convert into prefetch dataset
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
TRAIN_SIZE = len(train_dataset)
STEPS_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE
VALIDATION_STEPS = 200 // BATCH_SIZE
train = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train = train.prefetch(buffer_size=AUTOTUNE)
test = test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# %%
# 3. Create image segmentation model
# 3.1. Use a pretrained model as the feature extraction layers
base_model = keras.applications.MobileNetV2(
    input_shape=[128, 128, 3], include_top=False
)

# 8.2. List down some activation layers
layer_names = [
    "block_1_expand_relu",  # 64x64
    "block_3_expand_relu",  # 32x32
    "block_6_expand_relu",  # 16x16
    "block_13_expand_relu",  # 8x8
    "block_16_project",  # 4x4
]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Define the feature extraction model
down_stack = keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

# Define the upsampling path
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),  # 32x32 -> 64x64
]


def unet_model(output_channels: int):
    inputs = layers.Input(shape=[128, 128, 3])
    # Apply functional API to construct U-Net
    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections(concatenation)
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model (output layer)
    last = layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2, padding="same"
    )  # 64x64 --> 128x128

    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)


# %%
# Make of use of the function to construct the entire U-Net
OUTPUT_CLASSES = 2

model = unet_model(output_channels=OUTPUT_CLASSES)
# Compile the model
loss = losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
keras.utils.plot_model(model, show_shapes=True)


# %%
def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()


for images, masks in train.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image, sample_mask])


# %%
# Create functions to show predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask


def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)[0]])
        else:
            display(
                [
                    sample_image,
                    sample_mask,
                    create_mask(model.predict(sample_image[tf.newaxis, ...]))[0],
                ]
            )


# %%
# Create a callback to help display results during model training
class DisplayCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print("\nSample Prediction after epoch {}\n".format(epoch + 1))


# %%


log_dir = os.path.join(
    os.getcwd(), "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)

EPOCH = 20

history = model.fit(
    train,
    epochs=EPOCH,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    validation_data=test,
    callbacks=[DisplayCallback(), tb_callback],
)

# %%
show_predictions(test, 3)
# %%
model_path = os.path.join(os.getcwd(), "model", "model.h5")
model.save(model_path)
