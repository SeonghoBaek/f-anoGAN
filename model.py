import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.utils import shuffle
import util
import layers
import argparse


def prepare_samples(base_dir, size_per_class):
    subdir_list = os.listdir(base_dir)

    classes = []
    files = []
    class_label = 0

    for subdir in subdir_list:
        class_dir = os.path.join(train_data, subdir).replace("\\", "/")

        files_in_class = os.listdir(class_dir)
        size = len(files_in_class)

        if size == 0:
            continue

        file_list = []
        num_itr = 0

        if size < size_per_class:
            num_itr = (size_per_class / size)

        residual = size_per_class - num_itr * size

        for i in range(num_itr):
            file_list = file_list + files_in_class

        file_list = file_list + files_in_class[:residual]

        for file in file_list:
            fullname_file = os.path.join(subdir, file).replace("\\", "/")
            files.append(fullname_file)
            classes.append(class_label)

        class_label = class_label + 1

    return files, classes


def load_images(file_name_list, base_dir, b_grayscale=False):
    images = []

    for file_name in file_name_list:
        fullname = os.path.join(base_dir, file_name).replace("\\", "/")

        img = cv2.imread(fullname)

        if b_grayscale is True:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img is not None:
            offset = input_width // 2
            # Center crop
            center_x = img_width // 2
            center_y = img_height // 2 + 4
            img = img[center_y - offset:center_y + offset, center_x - offset:center_x + offset]
            img = img * 1.0
            n_img = (img - 127.5) / 127.5
            images.append(n_img)

    images = np.array(images)

    if b_grayscale is True:
        images = np.expand_dims(images, axis=-1)

    return images


def generator(latent, category, activation='swish', scope='generator_network', norm='layer', b_train=False):
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
        l = layers.fc(l, 6 * 6 * unit_block_depth*2, non_linear_fn=act_func, scope='fc1', use_bias=False)
        print(' FC1: ' + str(l.get_shape().as_list()))

        l = tf.reshape(l, shape=[-1, 6, 6, unit_block_depth*2])

        # Init Stage. Coordinated convolution: Embed explicit positional information
        block_depth = unit_block_depth * 16
        l = layers.conv(l, scope='init', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_init')
        l = act_func(l)

        upsample_num_itr = 4
        for i in range(upsample_num_itr):
            # ESPCN upsample
            block_depth = block_depth // 2
            l = layers.conv(l, scope='upsample_' + str(i), filter_dims=[3, 3, block_depth * 2 * 2],
                            stride_dims=[1, 1], non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='upsample_norm_' + str(i))
            l = act_func(l)
            l = tf.nn.depth_to_space(l, 2)
            print(' Upsample Block : ' + str(l.get_shape().as_list()))

        # Bottleneck stage
        for i in range(bottleneck_depth):
            print(' Bottleneck Block : ' + str(l.get_shape().as_list()))
            l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                             norm=norm, b_train=b_train, scope='bt_block_' + str(i))

        # Transform to input channels
        l = layers.conv(l, scope='last', filter_dims=[3, 3, num_channel], stride_dims=[1, 1],
                        non_linear_fn=tf.nn.tanh,
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

        num_iter = 4

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

        num_encoder_feature_blocks = 5
        num_encoder_bt_blocks = 4

        print('Encoder Input: ' + str(x.get_shape().as_list()))
        block_depth = unit_block_depth

        l = layers.conv(x, scope='conv0', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm0')
        l = act_func(l)

        print('Encoder Block: ' + str(l.get_shape().as_list()))

        for i in range(num_encoder_bt_blocks):
            l = layers.add_se_residual_block(l,  filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                             norm=norm, b_train=b_train, scope='res_block_' + str(i))

            print('Encoder Bottleneck Block ' + str(i) + ': ' + str(l.get_shape().as_list()))
            block_depth = block_depth * 2
            l = layers.conv(l, scope='tr_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_' + str(i))
            l = act_func(l)

        last_layer = l

        latent = layers.global_avg_pool(last_layer, output_length=representation_dim, use_bias=False, scope='gp')
        categories = layers.fc(latent, num_class)

        print('Encoder Latent Dims: ' + str(latent.get_shape().as_list()))

    return latent, categories


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
    elif type == 'ce':
        eps = 1e-10
        loss = tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
    elif type == 'l1':
        loss = tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        loss = tf.reduce_mean(tf.square(tf.subtract(target, value)))

    loss = gamma * loss

    return loss


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


def generate_sample_z(num_samples, sample_length):
    noise = np.random.randn(num_samples * sample_length)
    noise = noise.reshape((num_samples, sample_length))

    return noise


def train_encoder(model_path, gan_model_path):
    print('Please wait. It takes several minutes. Do not quit!')

    with tf.device('/device:CPU:0'):
        X = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        C = tf.placeholder(tf.float32, [batch_size, num_class])
        LR = tf.placeholder(tf.float32, None)  # Learning Rate
        b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    latent, category = encoder(X, activation='lrelu', norm='instance', b_train=b_train, scope='encoder')
    fake_X = generator(latent, C, activation='relu', norm='batch', b_train=b_train, scope='generator')

    feature_fake, logit_fake = discriminator(fake_X, C, activation='lrelu', norm='instance', b_train=b_train,
                                             scope='discriminator')
    feature_real, logit_real = discriminator(X, C, activation='lrelu', norm='instance', b_train=b_train,
                                             scope='discriminator')

    feature_loss = get_residual_loss(feature_real, feature_fake, type='l2')
    encoder_loss = get_residual_loss(X, fake_X, type='l1')
    class_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=C, logits=category))
    encoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')

    alpha = 1.0
    weight_decay = 1e-5
    l2_regularizer = tf.add_n([tf.nn.l2_loss(v) for v in encoder_vars if 'bias' not in v.name])
    total_loss = encoder_loss + class_loss + weight_decay * l2_regularizer + alpha * feature_loss

    encoder_optimizer = tf.train.RMSPropOptimizer(learning_rate=LR).minimize(total_loss, var_list=encoder_vars)

    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
    generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        gan_variables_to_restore = disc_vars + generator_vars
        encoder_variables_to_restore = encoder_vars

        try:
            encoder_saver = tf.train.Saver(var_list=encoder_variables_to_restore)
            encoder_saver.restore(sess, model_path)
            gan_saver = tf.train.Saver(var_list=gan_variables_to_restore)
            gan_saver.restore(sess, gan_model_path)

            print('Model Restored')
        except:
            try:
                gan_saver = tf.train.Saver(var_list=gan_variables_to_restore)
                gan_saver.restore(sess, gan_model_path)
                print('Start New Training. Wait ...')
            except:
                print('Model load failed. No Pretrained GAN.')
                return

        file_list, classes = prepare_samples(train_data, num_samples_per_class)
        trX = file_list

        print('Number of Training Images: ' + str(len(trX)))
        total_input_size = len(trX) // batch_size
        total_steps = (total_input_size * num_epoch)
        learning_rate = 2e-4
        category_index = np.eye(num_class)[np.arange(num_class)]

        for e in range(num_epoch):
            trX, classes = shuffle(trX, classes)
            training_batch = zip(range(0, len(trX), batch_size),
                                 range(batch_size, len(trX) + 1, batch_size))
            itr = 0

            for start, end in training_batch:
                imgs = load_images(trX[start:end], base_dir=train_data, b_grayscale=use_gray_scale)
                categories = category_index[classes[start:end]]
                cur_steps = (e * total_input_size) + itr + 1.0
                lr = learning_rate * np.cos((np.pi * 7 / 16) * (cur_steps / total_steps))

                _, e_loss, c_loss, f_loss, decoded_images = sess.run([encoder_optimizer, encoder_loss, class_loss, feature_loss, fake_X],
                                                                     feed_dict={X: imgs, C: categories, b_train: True, LR: lr})

                print('epoch: ' + str(e) + ', e loss: ' + str(e_loss) + ', c loss: ' + str(c_loss) + ', f loss: ' + str(f_loss))

                for i in range(batch_size):
                    if use_gray_scale is True:
                        img = cv2.cvtColor(decoded_images[i] * 127.5 + 127.5, cv2.COLOR_GRAY2BGR)
                    else:
                        img = decoded_images[i] * 127.5 + 127.5
                    cv2.imwrite('samples_encoder/' + trX[start + i], img)

                itr += 1

                if itr % 200 == 0:
                    try:
                        print('Saving model...')
                        saver = tf.train.Saver(encoder_vars)
                        saver.save(sess, model_path)
                        print('Saved.')
                    except:
                        print('Save failed')

            try:
                print('Saving model...')
                saver = tf.train.Saver(encoder_vars)
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
    fake_X = generator(Z, C, activation='relu', norm='batch', b_train=b_train, scope='generator')
    augmented_fake_X = util.random_augments(fake_X)

    # Adversarial Discriminator
    augmented_X = util.random_augments(X)
    feature_real, logit_real = discriminator(augmented_X, C, activation='lrelu', norm='instance', b_train=b_train, scope='discriminator')
    feature_fake, logit_fake = discriminator(augmented_fake_X, C, activation='lrelu', norm='instance', b_train=b_train, scope='discriminator')

    grad_loss = get_gradient_loss(X, fake_X)
    feature_loss = get_feature_matching_loss(feature_real, feature_fake, type='l2')

    label_smooth_real = tf.ones_like(logit_real) - 0.3 + tf.random_uniform([], minval=0.0, maxval=1.0, dtype=tf.float32) * 0.5
    disc_loss = get_discriminator_loss(logit_real, label_smooth_real, type='ls') + \
                get_discriminator_loss(logit_fake, tf.zeros_like(logit_fake), type='ls')

    gen_ls_loss = get_discriminator_loss(logit_fake, tf.ones_like(logit_fake), type='ls')
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    gen_l2_regularizer = tf.add_n([tf.nn.l2_loss(v) for v in generator_vars if 'bias' not in v.name])
    weight_decay = 1e-5
    gen_loss = gen_ls_loss

    disc_optimizer = tf.train.RMSPropOptimizer(learning_rate=LR).minimize(disc_loss, var_list=disc_vars)
    gen_optimizer = tf.train.RMSPropOptimizer(learning_rate=LR).minimize(gen_loss, var_list=generator_vars)

    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model Restored')
        except:
            print('Start New Training. Wait ...')

        file_list, classes = prepare_samples(train_data, num_samples_per_class)

        trX = file_list

        print('Number of Training Images: ' + str(len(trX)))

        # How many times  Discriminator is updated per 1 Generator update.
        num_critic = 3
        learning_rate = 2e-4

        category_index = np.eye(num_class)[np.arange(num_class)]

        total_input_size = len(trX) // batch_size
        total_steps = (total_input_size * num_epoch)

        for e in range(num_epoch):
            trX, classes = shuffle(trX, classes)
            training_batch = zip(range(0, len(trX), batch_size),  range(batch_size, len(trX)+1, batch_size))
            itr = 0

            for start, end in training_batch:
                imgs = load_images(trX[start:end], base_dir=train_data, b_grayscale=use_gray_scale)

                categories = category_index[classes[start:end]]
                noise = generate_sample_z( num_samples=batch_size, sample_length=representation_dim)

                cur_steps = (e * total_input_size) + itr + 1.0
                lr = learning_rate * np.cos((np.pi * 7 / 16) * (cur_steps / total_steps))

                _, d_loss = sess.run([disc_optimizer, disc_loss],
                                     feed_dict={Z: noise,
                                                X: imgs,
                                                C: categories,
                                                LR: lr,
                                                b_train: True})

                if itr % num_critic == 0:
                    # Separate Batch of Generator & Discriminator
                    noise = generate_sample_z(num_samples=batch_size, sample_length=representation_dim)

                    _, g_loss = sess.run([gen_optimizer, gen_loss],
                                         feed_dict={Z: noise,
                                                    X: imgs,
                                                    C: categories,
                                                    LR: lr,
                                                    b_train: True})

                    decoded_images = sess.run([fake_X], feed_dict={Z: noise, C: categories, b_train: True})
                    print('epoch: ' + str(e) + ', discriminator: ' + str(d_loss) +
                          ', generator: ' + str(g_loss))

                    decoded_images = decoded_images[0]

                    if use_gray_scale is True:
                        img = cv2.cvtColor(decoded_images[0] * 127.5 + 127.5, cv2.COLOR_GRAY2BGR)
                    else:
                        img = decoded_images[0] * 127.5 + 127.5
                    cv2.imwrite(train_out_dir + '/' + str(itr) + '_out.png', img)

                itr += 1

                if itr % 200 == 0:
                    try:
                        print('Saving model...')
                        saver.save(sess, model_path)
                        print('Saved.')
                    except:
                        print('Save failed')

            try:
                print('Saving model...')
                saver.save(sess, model_path)
                print('Saved.')
            except:
                print('Save failed')


def test(model_path):
    print('Please wait. It takes several minutes. Do not quit!')

    with tf.device('/device:CPU:0'):
        Z = tf.placeholder(tf.float32, [batch_size, representation_dim])
        C = tf.placeholder(tf.float32, [batch_size, num_class])
        b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    # Content discriminator
    fake_X = generator(Z, C, activation='relu', norm='batch', b_train=b_train, scope='generator')
    generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver(var_list=generator_vars)
            saver.restore(sess, model_path)
            print('Model Restored')
        except:
            print('Start New Training. Wait ...')

        category_index = np.eye(num_class)[np.arange(num_class)]

        num_itr = num_test_output // batch_size
        for i in range(num_class):
            cls = [i] * batch_size
            categories = category_index[cls]

            for j in range(num_itr):
                noise = generate_sample_z(num_samples=batch_size, sample_length=representation_dim)
                decoded_images = sess.run([fake_X], feed_dict={Z: noise, C: categories, b_train: False})

                decoded_images = decoded_images[0]

                for k in range(batch_size):
                    if use_gray_scale is True:
                        img = cv2.cvtColor(decoded_images[k] * 127.5 + 127.5, cv2.COLOR_GRAY2BGR)
                    else:
                        img = decoded_images[k] * 127.5 + 127.5

                    cv2.imwrite(test_out_dir + '/' + str(i) + '_' + str(j) + '_' + str(k) + '.png', img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train_gan/train_encoder/test', default='train_gan')
    parser.add_argument('--model_path', type=str, help='model check point file path', default='./model/m.ckpt')
    parser.add_argument('--gan_model_path', type=str, help='gan model check point file path', default='./model_gan/m.ckpt')
    parser.add_argument('--train_data', type=str, help='training data directory', default='data')
    parser.add_argument('--train_out_dir', type=str, help='train output directory', default='train_out')
    parser.add_argument('--test_out_dir', type=str, help='test output directory', default='test_out')

    args = parser.parse_args()

    train_data = args.train_data
    train_out_dir = args.train_out_dir
    test_out_dir = args.test_out_dir
    model_path = args.model_path
    gan_model_path = args.gan_model_path

    # Source image size
    img_width = 128
    img_height = 128
    # Network input size
    input_width = 96
    input_height = 96
    num_channel = 3
    use_gray_scale = True
    num_samples_per_class = 744

    if use_gray_scale is True:
        num_channel = 1

    # Network Configurations
    unit_block_depth = 32
    bottleneck_depth = 8
    batch_size = 64
    representation_dim = 128
    num_epoch = 10000
    num_class = 12

    if args.mode == 'train_gan':
        train_gan(model_path)
    elif args.mode == 'train_encoder':
        train_encoder(model_path, gan_model_path)
    elif args.mode == 'test_gan':
        batch_size = 64
        num_test_output = batch_size * 2
        test(model_path)
