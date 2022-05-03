import tensorflow as tf

import keras
import keras.backend as K
# inspired by https://github.com/keras-team/keras/issues/9395


def diceNecrotic(y_true, y_pred, epsilon=1e-6):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    intersection = K.sum(K.abs(y_true[:, :, :, 1] * y_pred[:, :, :, 1]))
    return (2. * intersection) / (K.sum(K.square(y_true[:, :, :, 1])) + K.sum(K.square(y_pred[:, :, :, 1])) + epsilon)


def diceEdema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, 2] * y_pred[:, :, :, 2]))
    return (2. * intersection) / (K.sum(K.square(y_true[:, :, :, 2])) + K.sum(K.square(y_pred[:, :, :, 2])) + epsilon)


def diceEnhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, 3] * y_pred[:, :, :, 3]))
    return (2. * intersection) / (K.sum(K.square(y_true[:, :, :, 3])) + K.sum(K.square(y_pred[:, :, :, 3])) + epsilon)


def diceLoss(y_true, y_pred, class_weights):
    y_true = tf.convert_to_tensor(y_true, 'float32')
    y_pred = tf.convert_to_tensor(y_pred, y_true.dtype)

    num = tf.math.reduce_sum(tf.math.multiply(class_weights, tf.math.reduce_sum(
        tf.math.multiply(y_true, y_pred), axis=[0, 1, 2, 3])))
    den = tf.math.reduce_sum(tf.math.multiply(class_weights, tf.math.reduce_sum(
        tf.math.add(y_true, y_pred), axis=[0, 1, 2, 3])))+1e-5

    return 1-2*num/den


def discriminator_loss(disc_real_output, disc_fake_output):
    real_loss = tf.math.reduce_mean(tf.math.pow(
        tf.ones_like(disc_real_output) - disc_real_output, 2))
    fake_loss = tf.math.reduce_mean(tf.math.pow(
        tf.zeros_like(disc_fake_output) - disc_fake_output, 2))

    disc_loss = 0.5*(real_loss + fake_loss)

    return disc_loss


def generator_loss(target, gen_output, disc_fake_output, class_weights, alpha):

    # generalized dice loss
    dice_loss = diceLoss(target, gen_output, class_weights)

    # disc loss
    disc_loss = tf.math.reduce_mean(tf.math.pow(
        tf.ones_like(disc_fake_output) - disc_fake_output, 2))

    # total loss
    gen_loss = alpha*dice_loss + disc_loss

    dice_necrotic = diceNecrotic(target, gen_output)
    dice_edema = diceEdema(target, gen_output)
    dice_enhancing = diceEnhancing(target, gen_output)
    print('test_step:  dice_necrotic, dice_edema, dice_enhancing')
    print(dice_necrotic, dice_edema, dice_enhancing)

    return gen_loss, dice_loss, disc_loss
