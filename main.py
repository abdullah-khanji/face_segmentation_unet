import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.test.is_built_with_cuda():
    print("The installed TensorFlow is built with CUDA")
tf.config.list_physical_devices('GPU')
