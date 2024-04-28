import tensorflow as tf

def conv_block(inputs, filters):
    c= tf.keras.layers.Conv2D(filters, 3, padding='same')(inputs)
    c= tf.keras.layers.BatchNormalization()(c)
    c= tf.keras.layers.Activation("relu")(c)

    c= tf.keras.layers.Conv2D(filters, 3, padding='same')(c)
    c= tf.keras.layers.BatchNormalization()(c)
    c= tf.keras.layers.Activation("relu")(c)

    return c

def encoder_block(inputs, filters):
    c= conv_block(inputs, filters)
    m= tf.keras.layers.MaxPool2D((2, 2))(c)
    return c, m


def decoder_block(inputs, c, filters):
    u= tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(inputs)
    u= tf.keras.layers.concatenate([u, c])
    u= conv_block(u, filters)
    return u

def build_unet(input_shape, num_classes):
    input= tf.keras.layers.Input(input_shape)

    c1, m1= encoder_block(input, 16)
    c2, m2= encoder_block(m1, 32)
    c3, m3= encoder_block(m2, 64)
    c4, m4= encoder_block(m3, 128)

    c5= conv_block(m4, 256)

    u6= decoder_block(c5, c4, 128)
    u7= decoder_block(u6, c3, 64)
    u8= decoder_block(u7, c2, 32)
    u9= decoder_block(u8, c1, 16)
    output= tf.keras.layers.Conv2D(num_classes, 1,activation='softmax')(u9)
    model= tf.keras.models.Model(input, output)
    return model

if __name__=="__main__":
    input_shape=(128, 128, 3)
    model= build_unet(input_shape, 11)
    model.summary()