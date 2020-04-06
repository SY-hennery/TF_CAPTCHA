#coding:utf-8
import numpy as np
import tensorflow as tf
import sys
#from captcha_creater import dataset
import captcha_creater as cc

DATASET_LEN = len(cc.dataset)

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4

# 申请三个占位符
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*DATASET_LEN])
keep_prob = tf.placeholder(tf.float32) # dropout

# 定义CNN
def create_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # 3 conv layer # 3 个 转换层
    w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer  # 最后连接层
    w_d = tf.Variable(w_alpha*tf.random_normal([8*20*64, 1024]))
    b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    # 输出层
    w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*DATASET_LEN]))
    b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*DATASET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


def captcha_training_start():
    cnn = create_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cnn, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    predict = tf.reshape(cnn, [-1, MAX_CAPTCHA, DATASET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, DATASET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('./model/'))
        step = 0
        while True:
            batch_x, batch_y = cc.create_batch(32)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})

            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = cc.create_batch(50)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print("step:%d  loss:%f  acc:%f" %(step, loss_ , acc))
                saver.save(sess, "./model/crack_capcha.model")
                if acc > 0.95:
                    break
                
            step += 1


def get_captcha(captcha_image):
    output = create_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./model/'))
        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, DATASET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
        text = text_list[0].tolist()
        vector = np.zeros(MAX_CAPTCHA*DATASET_LEN)
        i = 0
        for n in text:
                vector[i*DATASET_LEN + n] = 1
                i += 1
        return cc.vec2text(vector)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        captcha_training_start()
    elif sys.argv[1] == 'train':
        captcha_training_start()
    elif sys.argv[1] == 'check':
        img,txt = cc.create_captcha()
        cc.parse_image(img)
        img = img.flatten() / 255 # 将图片一维化
        ans = get_captcha(img)
        print('captcha predict::   expected:%s  real:%s' %(txt,ans))
    else:
        captcha_training_start()