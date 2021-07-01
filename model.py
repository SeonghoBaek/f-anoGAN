import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.utils import shuffle
import util
import layers
import argparse


def load_images(file_name_list, base_dir, use_augmentation=False, add_eps=False, rotate=-1, resize=[256, 256]):
    images = []
    ca = []

    for file_name in file_name_list:
        fullname = os.path.join(base_dir, file_name).replace("\\", "/")
        img = cv2.imread(fullname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # resized_img = cv2.resize(img, dsize=(512, 520), interpolation=cv2.INTER_CUBIC)
        # resized_img_hd = cv2.resize(img, dsize=(1024, 1040), interpolation=cv2.INTER_CUBIC)

        if img is not None:
            #print(fullname)
            #img = img[4:input_height, 0:input_width]

            # Center crop
            center_x = img_width // 2
            center_y = img_height // 2
            img = img[center_y - 48:center_y + 48, center_x - 48:center_x + 48]

            if use_augmentation is True:
                aug = np.random.randint(low=0, high=4)

                if aug == 0:
                    img = cv2.flip(img, 1)
                elif aug == 1:
                    rotate = np.random.randint(low=0, high=3)
                    img = cv2.rotate(img, rotate)
                elif aug == 2:
                    img = img + np.random.uniform(low=0, high=1, size=img.shape)

            #img = cv2.resize(img, dsize=(input_width, input_height), interpolation=cv2.INTER_LINEAR)
            n_img = img / 255.0
            images.append(n_img)
            ca.append(int(file_name[0]))

    return np.array(images), np.array(ca)


def generator(latent, category, activation='swish', scope='generator_network', norm='layer', b_train=False, use_upsample=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        print(scope + ' Input: ' + str(latent.get_shape().as_list()))

        l = tf.concat([latent, category], axis=-1)
        print(' Concat category: ' + str(l.get_shape().as_list()))
        l = layers.fc(l, 48 * 48 * num_channel, non_linear_fn=act_func, scope='fc1', use_bias=False)
        print(' FC1: ' + str(l.get_shape().as_list()))

        l = tf.reshape(l, shape=[-1, 48, 48, num_channel])

        # Init Stage. Coordinated convolution: Embed explicit positional information
        block_depth = unit_block_depth * 2
        l = layers.conv(l, scope='init', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_init')
        l = act_func(l)

        upsample_num_itr = 1
        for i in range(upsample_num_itr):
            # ESPCN upsample
            block_depth = block_depth // 2
            l = layers.conv(l, scope='espcn_' + str(i), filter_dims=[3, 3, block_depth * 2 * 2],
                            stride_dims=[1, 1], non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='espcn_norm_' + str(i))
            l = act_func(l)
            l = tf.nn.depth_to_space(l, 2)

        # Bottleneck stage
        for i in range(bottleneck_depth):
            print(' Bottleneck Block : ' + str(l.get_shape().as_list()))
            l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                             norm=norm, b_train=b_train, use_residual=True, use_dilation=False,
                                             scope='bt_block_' + str(i))

        # Transform to input channels
        l = layers.conv(l, scope='last', filter_dims=[3, 3, num_channel], stride_dims=[1, 1],
                        non_linear_fn=tf.nn.sigmoid,
                        bias=False)

    print('Generator Output: ' + str(l.get_shape().as_list()))

    return l


def discriminator(x, category, activation='relu', scope='discriminator_network', norm='layer', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        print(scope + ' Input: ' + str(x.get_shape().as_list()))

        block_depth = unit_block_depth
        norm_func = norm
        b, h, w, c = x.get_shape().as_list()

        cat = tf.reshape(category, shape=[-1, 1, 1, num_class])
        l = tf.concat([x, cat * tf.ones([b, h, w, num_class])], -1)
        print(scope + 'Concat Input: ' + str(l.get_shape().as_list()))

        l = layers.conv(l, scope='init', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False, padding='SAME')
        l = layers.conv_normalize(l, norm=norm_func, b_train=b_train, scope='norm_init')
        l = act_func(l)

        num_iter = 3

        for i in range(num_iter):
            l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                             norm=norm_func, b_train=b_train, scope='disc_block_1_' + str(i))
            block_depth = block_depth * 2
            l = layers.conv(l, scope='dn_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                            non_linear_fn=None, bias=False)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='dn_norm_' + str(i))
            l = act_func(l)

        print('Discriminator Block : ' + str(l.get_shape().as_list()))

        last_layer = l
        logit = layers.conv(last_layer, scope='conv_pred', filter_dims=[3, 3, 1], stride_dims=[1, 1],
                            non_linear_fn=None, bias=False)

        print('Discriminator Logit Dims: ' + str(logit.get_shape().as_list()))

    return last_layer, logit


def encoder(x, activation='relu', scope='encoder', norm='layer', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        print('Encoder Input: ' + str(x.get_shape().as_list()))
        block_depth = dense_block_depth

        # [128 x 128]
        l = layers.conv(x, scope='conv0', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm0')
        l = act_func(l)

        print('Encoder Block 0: ' + str(l.get_shape().as_list()))

        l = layers.add_residual_dense_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                            act_func=act_func, norm=norm, b_train=b_train, scope='dense_block_1')

        block_depth = block_depth * 2

        # [64 x 64]
        l = layers.conv(l, scope='tr1', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm1')
        l = act_func(l)

        print('Encoder Block 1: ' + str(l.get_shape().as_list()))

        for i in range(2):
            l = layers.add_residual_block(l,  filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                          norm=norm, b_train=b_train, scope='res_block_1_' + str(i))

        block_depth = block_depth * 2

        # [32 x 32]
        l = layers.conv(l, scope='tr2', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm2')
        l = act_func(l)

        print('Encoder Block 2: ' + str(l.get_shape().as_list()))

        for i in range(2):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                          norm=norm, b_train=b_train, use_dilation=False, scope='res_block_2_' + str(i))
        block_depth = block_depth * 2

        # [16 x 16]
        l = layers.conv(l, scope='tr3', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm3')
        l = act_func(l)

        print('Encoder Block 3: ' + str(l.get_shape().as_list()))

        for i in range(2):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                          norm=norm, b_train=b_train, use_dilation=False, scope='res_block_3_' + str(i))

        # [8 x 8]
        block_depth = block_depth * 2
        l = layers.conv(l, scope='tr4', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm4')
        l = act_func(l)

        print('Encoder Block 4: ' + str(l.get_shape().as_list()))

        for i in range(2):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                          norm=norm, b_train=b_train, use_dilation=False, scope='res_block_4_' + str(i))

        # [4 x 4]
        block_depth = block_depth * 2
        l = layers.conv(l, scope='tr5', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm5')
        l = act_func(l)

        print('Encoder Block 5: ' + str(l.get_shape().as_list()))

        for i in range(2):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                          norm=norm, b_train=b_train, use_dilation=False,
                                          scope='res_block_5_' + str(i))

        last_layer = l

        context = layers.global_avg_pool(last_layer, output_length=representation_dim, use_bias=False, scope='gp')

        print('Encoder GP Dims: ' + str(context.get_shape().as_list()))

    return context


def latent_discriminator(x, activation='relu', scope='latent_discriminator_network', norm='layer', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        print('Latent Discriminator Input: ' + str(x.get_shape().as_list()))

        l = x
        l = layers.fc(l, l.get_shape().as_list()[-1] // 2, non_linear_fn=act_func, scope='flat1')
        print('Latent Discriminator layer 1: ' + str(l.get_shape().as_list()))
        #l = layers.layer_norm(l, scope='ln0')

        feature = layers.fc(l, l.get_shape().as_list()[-1] // 4, non_linear_fn=act_func, scope='flat2')
        print('Latent Discriminator Feature: ' + str(feature.get_shape().as_list()))
        logit = layers.fc(feature, 1, non_linear_fn=None, scope='final')

    return feature, logit


def get_feature_matching_loss(value, target, type='l1', gamma=1.0):
    if type == 'rmse':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))
    elif type == 'cross-entropy':
        eps = 1e-10
        loss = tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
    elif type == 'l1':
        loss = tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        loss = tf.reduce_mean(tf.square(tf.subtract(target, value)))
        #loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(target, value)))))
    return gamma * loss


def get_discriminator_loss(real, fake, type='wgan', gamma=1.0):
    if type == 'wgan':
        # wgan loss
        d_loss_real = tf.reduce_mean(real)
        d_loss_fake = tf.reduce_mean(fake)

        # W Distant: f(real) - f(fake). Maximizing W Distant.
        return gamma * (d_loss_fake - d_loss_real), d_loss_real, d_loss_fake
    elif type == 'ce':
        # cross entropy
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        return gamma * (d_loss_fake + d_loss_real), d_loss_real, d_loss_fake
    elif type == 'hinge':
        d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - real))
        d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + fake))
        return gamma * (d_loss_fake + d_loss_real), d_loss_real, d_loss_fake
    elif type == 'ls':
        return tf.reduce_mean((real - fake) ** 2)


def get_residual_loss(value, target, type='l1', gamma=1.0):
    if type == 'rmse':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))
    elif type == 'cross-entropy':
        eps = 1e-10
        loss = tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
    elif type == 'l1':
        #loss = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(target, value)), [1]))
        loss = tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        #loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(target, value)), [1]))
        loss = tf.reduce_mean(tf.square(tf.subtract(target, value)))

    loss = gamma * loss

    return loss


def get_diff_loss(anchor, positive, negative):
    a_p = get_residual_loss(anchor, positive, 'l1')
    a_n = get_residual_loss(anchor, negative, 'l1')
    # a_n > a_p + margin
    # a_p - a_n + margin < 0
    # minimize (a_p - a_n + margin)
    return tf.reduce_mean(a_p / a_n)


def get_gradient_loss(img1, img2):
    # Laplacian second derivation
    image_a = img1  # tf.expand_dims(img1, axis=0)
    image_b = img2  # tf.expand_dims(img2, axis=0)

    dx_a, dy_a = tf.image.image_gradients(image_a)
    dx_b, dy_b = tf.image.image_gradients(image_b)

    '''
    d2x_ax, d2y_ax = tf.image.image_gradients(dx_a)
    d2x_bx, d2y_bx = tf.image.image_gradients(dx_b)
    d2x_ay, d2y_ay = tf.image.image_gradients(dy_a)
    d2x_by, d2y_by = tf.image.image_gradients(dy_b)

    loss1 = tf.reduce_mean(tf.abs(tf.subtract(d2x_ax, d2x_bx))) + tf.reduce_mean(tf.abs(tf.subtract(d2y_ax, d2y_bx)))
    loss2 = tf.reduce_mean(tf.abs(tf.subtract(d2x_ay, d2x_by))) + tf.reduce_mean(tf.abs(tf.subtract(d2y_ay, d2y_by)))
    '''
    loss1 = tf.reduce_mean(tf.abs(tf.subtract(dx_a, dx_b)))
    loss2 = tf.reduce_mean(tf.abs(tf.subtract(dy_a, dy_b))) 

    return (loss1+loss2)


def generate_sample_z(low, high, num_samples, sample_length, b_uniform=True):
    samples = []

    for i in range(num_samples):
        if b_uniform is True:
            noise = np.random.uniform(low=low, high=high, size=[sample_length])
        else:
            noise = np.random.normal(low, high, size=[sample_length])
        samples.append(noise)

    return np.array(samples)


def make_multi_modal_noise(num_mode=8):
    size = representation_dim // num_mode

    noise = tf.random_normal(shape=[batch_size, size], mean=0.0, stddev=1.0, dtype=tf.float32)

    for i in range(num_mode-1):
        n = tf.random_normal(shape=[batch_size, size], mean=0.0, stddev=1.0, dtype=tf.float32)
        noise = tf.concat([noise, n], axis=1)

    return noise


def train_encoder(model_path):
    print('Please wait. It takes several minutes. Do not quit!')

    with tf.device('/device:CPU:0'):
        X = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])

    b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    latent = encoder(X, activation='relu', norm='instance', b_train=b_train, scope='encoder')
    fake_X = generator(latent, activation='relu', norm='instance', b_train=b_train, scope='generator',
                       use_upsample=True)
    # Adversarial Discriminator
    feature_fake, logit_fake = discriminator(fake_X, activation='swish', norm='instance', b_train=b_train,
                                             scope='discriminator', use_patch=False)
    feature_real, logit_real = discriminator(X, activation='swish', norm='instance', b_train=b_train,
                                             scope='discriminator', use_patch=False)

    feature_loss = get_residual_loss(feature_real, feature_fake, type='l2')
    encoder_loss = get_residual_loss(X, fake_X, type='l2')
    encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')

    alpha = 1.0
    total_loss = encoder_loss + alpha * feature_loss

    encoder_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(total_loss, var_list=[encoder_vars])

    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model Restored')
        except:
            try:
                variables_to_restore = [v for v in tf.trainable_variables()
                                        if v.name.split('/')[0] == 'generator'
                                        or v.name.split('/')[0] == 'discriminator']
                saver = tf.train.Saver(variables_to_restore)
                print('Start New Training. Wait ...')
            except:
                print('Model load failed. No Pretrained GAN.')
                return

        trX = os.listdir(train_data)
        print('Number of Training Images: ' + str(len(trX)))
        num_augmentations = 1  # How many augmentations per 1 sample
        file_batch_size = batch_size // num_augmentations

        for e in range(num_epoch):
            trX = shuffle(trX)
            training_batch = zip(range(0, len(trX), file_batch_size),
                                 range(file_batch_size, len(trX) + 1, file_batch_size))
            itr = 0

            for start, end in training_batch:
                imgs = load_images(trX[start:end], base_dir=train_data, use_augmentation=False)
                #imgs = np.expand_dims(imgs, axis=3)

                _, e_loss, decoded_images = sess.run([encoder_optimizer, total_loss, fake_X], feed_dict={X: imgs, b_train: True})

                print('epoch: ' + str(e) + ', loss: ' + str(e_loss))

                decoded_images = np.squeeze(decoded_images)
                cv2.imwrite('imgs/' + trX[start], decoded_images[3] * 255)
                # cv2.imwrite('imgs/org_' + trX[start], imgs[3] * 255)

                itr += 1

                if itr % 200 == 0:
                    try:
                        print('Saving model...')
                        saver.save(sess, model_path)
                        print('Saved.')
                    except:
                        print('Save failed')

                if is_exit() is True:
                    return

            try:
                print('Saving model...')
                saver.save(sess, model_path)
                print('Saved.')
            except:
                print('Save failed')


def train_gan(model_path):
    print('Please wait. It takes several minutes. Do not quit!')

    with tf.device('/device:CPU:0'):
        X = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        Z = tf.placeholder(tf.float32, [batch_size, representation_dim])
        C = tf.placeholder(tf.float32, [batch_size, num_class])
        LR = tf.placeholder(tf.float32, None)  # Learning Rate

    b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    # Content discriminator
    fake_X = generator(Z, C, activation='relu', norm='instance', b_train=b_train, scope='generator', use_upsample=True)
    augmented_fake_X = util.random_augments(fake_X)

    # Adversarial Discriminator
    augmented_X = util.random_augments(X)
    feature_real, logit_real = discriminator(augmented_X, C, activation='relu', norm='instance', b_train=b_train, scope='discriminator')
    feature_fake, logit_fake = discriminator(augmented_fake_X, C, activation='relu', norm='instance', b_train=b_train, scope='discriminator')

    grad_loss = get_gradient_loss(X, fake_X)
    feature_loss = get_feature_matching_loss(feature_real, feature_fake, type='l2')

    #disc_loss, disc_loss_real, disc_loss_fake = get_discriminator_loss(logit_real, logit_fake, type='wgan')
    disc_loss = get_discriminator_loss(logit_real, tf.ones_like(logit_real), type='ls') + \
                get_discriminator_loss(logit_fake, tf.zeros_like(logit_fake), type='ls')

    #gen_loss = -tf.reduce_mean(logit_fake)
    gen_loss = get_discriminator_loss(logit_fake, tf.ones_like(logit_fake), type='ls') # + feature_loss #+ 0.1 * grad_loss
    #gen_loss = feature_loss

    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    disc_l2_regularizer = tf.add_n([tf.nn.l2_loss(v) for v in disc_vars if 'bias' not in v.name])
    weight_decay = 1e-5
    disc_loss = disc_loss + weight_decay * disc_l2_regularizer

    generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    gen_l2_regularizer = tf.add_n([tf.nn.l2_loss(v) for v in generator_vars if 'bias' not in v.name])
    gen_loss = gen_loss + weight_decay * gen_l2_regularizer

    disc_optimizer = tf.train.RMSPropOptimizer(learning_rate=LR).minimize(disc_loss, var_list=disc_vars)
    gen_optimizer = tf.train.RMSPropOptimizer(learning_rate=LR).minimize(gen_loss, var_list=generator_vars)
    #disc_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(disc_loss, var_list=disc_vars)
    #gen_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(gen_loss, var_list=generator_vars)
    image_pool = util.ImagePool(maxsize=30, threshold=0.5)

    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model Restored')
        except:
            print('Start New Training. Wait ...')

        trX = os.listdir(train_data)
        print('Number of Training Images: ' + str(len(trX)))
        num_augmentations = 1  # How many augmentations per 1 sample
        file_batch_size = batch_size // num_augmentations
        num_critic = 3
        learning_rate = 2e-4

        category_index = np.eye(num_class)[np.arange(num_class)]

        total_input_size = len(trX) // batch_size
        total_steps = (total_input_size * num_epoch)

        for e in range(num_epoch):
            trX = shuffle(trX)
            training_batch = zip(range(0, len(trX), file_batch_size),  range(file_batch_size, len(trX)+1, file_batch_size))
            itr = 0

            for start, end in training_batch:
                imgs, cats = load_images(trX[start:end], base_dir=train_data)
                imgs = np.expand_dims(imgs, axis=-1)
                #print(imgs.shape)

                categories = category_index[cats]
                noise = generate_sample_z(low=0, high=1.0, num_samples=batch_size,
                                          sample_length=representation_dim,
                                          b_uniform=False)

                cur_steps = (e * total_input_size) + itr + 1.0

                lr = learning_rate * np.cos((np.pi * 7 / 16) * (cur_steps / total_steps))

                _, d_loss = sess.run([disc_optimizer, disc_loss],
                                     feed_dict={Z: noise,
                                                X: imgs,
                                                C: categories,
                                                LR: lr,
                                                b_train: True})

                if itr % num_critic == 0:
                    _, g_loss = sess.run([gen_optimizer, gen_loss],
                                         feed_dict={Z: noise,
                                                    X: imgs,
                                                    C: categories,
                                                    b_train: True, LR: lr})

                    decoded_images = sess.run([fake_X], feed_dict={Z: noise, C: categories, b_train: True})
                    print('epoch: ' + str(e) + ', discriminator: ' + str(d_loss) +
                          ', generator: ' + str(g_loss))

                    decoded_images = np.squeeze(decoded_images)
                    #imgs = np.squeeze(imgs)

                    for i in range(batch_size):
                        cv2.imwrite('imgs/gen_' + trX[start+i], decoded_images[i] * 255.0)
                        #cv2.imwrite('imgs/' + trX[start+i], imgs[i] * 255.0)

                itr += 1

                if itr % 200 == 0:
                    try:
                        print('Saving model...')
                        saver.save(sess, model_path)
                        print('Saved.')
                    except:
                        print('Save failed')

                if is_exit() is True:
                    return

            try:
                print('Saving model...')
                saver.save(sess, model_path)
                print('Saved.')
            except:
                print('Save failed')


def test(model_path):
    with tf.device('/device:CPU:0'):
        X = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])

    b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    latent = encoder(X, activation='swish', norm='layer', b_train=b_train, scope='encoder')
    fake_X = generator(latent, activation='swish', norm='layer', b_train=b_train, scope='generator',
                       use_upsample=True)
    # Adversarial Discriminator
    feature_fake, logit_fake = discriminator(fake_X, activation='swish', norm='layer', b_train=b_train,
                                             scope='discriminator', use_patch=False)
    feature_real, logit_real = discriminator(X, activation='swish', norm='layer', b_train=b_train,
                                             scope='discriminator', use_patch=False)

    feature_loss = get_residual_loss(feature_real, feature_fake, type='l2')
    encoder_loss = get_residual_loss(X, fake_X, type='l1')
    encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')

    alpha = 1.0
    total_loss = encoder_loss + alpha * feature_loss

    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model Restored')
        except:
            print('Model Restore Failed')
            return

        trX = os.listdir(test_data)
        print('Number of Test Images: ' + str(len(trX)))
        file_batch_size = 1
        training_batch = zip(range(0, len(trX), file_batch_size),  range(file_batch_size, len(trX)+1, file_batch_size))

        for start, end in training_batch:
            imgs = load_images(trX[start:end], base_dir=test_data)
            #imgs = np.expand_dims(imgs, axis=3)
            #print('Image batch Shape: ' + str(imgs.shape))
            aae_loss = sess.run([total_loss], feed_dict={X: imgs, b_train: False})

            print(str(trX[start:end]) + ' score: ' + str(aae_loss))


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def is_exit(marker='.exit'):
    import os.path

    if os.path.isfile(marker) is True:
        os.remove(marker)
        return True

    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train_gan/train_encoder/test', default='train_gan')
    parser.add_argument('--model_path', type=str, help='model check point file path', default='./model/m.ckpt')
    parser.add_argument('--train_data', type=str, help='training data directory', default='input')
    parser.add_argument('--test_data', type=str, help='test data directory', default='test')

    args = parser.parse_args()

    train_data = args.train_data
    test_data = args.test_data
    model_path = args.model_path

    img_width = 128
    img_height = 128
    input_width = 96
    input_height = 96
    num_channel = 1

    unit_block_depth = 64
    dense_block_depth = 32

    # Bottle neck(depth narrow down) depth. See Residual Dense Block and Residual Block.
    bottleneck_depth = 12
    batch_size = 32
    representation_dim = 1024

    test_size = 100
    num_epoch = 300000
    num_class = 9

    if args.mode == 'train_gan':
        train_gan(model_path)
    elif args.mode == 'train_encoder':
        train_encoder(model_path)
    else:
        model_path = args.model_path
        test_data = args.test_data
        batch_size = 1
        test(model_path)
