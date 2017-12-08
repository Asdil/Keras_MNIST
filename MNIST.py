import keras
from keras.models import Sequential  # 序贯模型，即神经网络是单向的连接
from keras.datasets import mnist  # 手写数据集，为60000张手写图片和10000张验证图片，第一次运行会自动重网上下载(翻墙可能更快)
from keras.layers import Dense, Dropout, Flatten
# 分别引用Dense全连接层，Dropout层(断开一些连接防止过拟合)， Flatten用于将卷积层压扁,方便全连接层输入
from keras.layers import Conv2D, MaxPool2D # Conv2D卷积层， MaxPool2D使用maxpool的池化层
# 池化层往往跟在卷积层后面， 将之前卷基层得到的特征图做一个聚合统计，也有降维的功能，减少计算量(理论上池化层有信息缺失)
from keras import backend as K # 这个是用于调节Theano和tensorflow输入
# Theano图像输入是channels_first也就是RGB通道在前面(3,28,28), 3表示红黄蓝三色的颜色通道， 28分别代表图片长宽像素
# tensorflow 相反(28,28,3)RGB通道在最后

# 预设参数
batch_size = 128  # 用于随机梯度下降，每次取128个样本作梯度下降，保证速度的情况下在最优点不至于剧烈震荡
num_classes = 10  # 有多少个类，这里手写体识别有10个类 类标签分类是0-9
epochs = 1   # 训练次数
# 指定输入数据的像素，因为是灰度图片所以没有颜色通道
img_rows, img_cols = 28, 28  # 这里现指定图片长宽
(x_train, y_train), (x_test, y_test) = mnist.load_data() # 加载数据集

# 设定输入格式，theano属于channel_first
# tensorflow属于channel_last
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    # 样本数 颜色通道  行数 列数
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)  # 这里是Theano的输入格式
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    # 样本数 颜色通道  行数 列数
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)  # 这里是tensorflow的输入格式

# 把数据变为float32
x_train = x_train.astype('float32')  # 将数据变为float32
x_test = x_test.astype('float32')
x_train /= 255  # 灰度值归一化
x_test /= 255
# 上面的步骤实际上是归一化图像数据，我们知道图像颜色范围在0-255之间，这里每个像素除以255相当于做了一个归一化
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'x_train samples')  # 输出训练样本和测试样本个数
print(x_test.shape[0], 'x_test samples')

# 把类别0-9变为二进制方便训练,softmax的输出
y_train = keras.utils.to_categorical(y_train, num_classes)
# 这里需要对标签做一个onehot编码，注意只有最后会后一层输出的激活层是softmax才使用
y_test = keras.utils.to_categorical(y_test, num_classes)

# 定义序贯模型
model = Sequential()
model.add(Conv2D(32, (3, 3),
                 activation='relu',
                 input_shape=input_shape))  # 卷积层有32个节点，每个卷积核的大小是(3,3)，使用relu作为激活函数
model.add(Conv2D(64, (3, 3), activation='relu'))  # 卷积层有64个节点

# 池化层
model.add(MaxPool2D(pool_size=(2, 2)))  # 将样本打个对折，这里的(2, 2)意思是将图片长宽/2, 如果是28，28的图片则变为14*14
model.add(Dropout(0.35))  # 断开一些连接，只保留0.35的节点连接，防止过拟合
model.add(Flatten())  # 将卷积层数据扁平化，变成一位向量
model.add(Dense(128, activation='relu'))  # 一个128个节点的全连接层
model.add(Dropout(0.5))  # 断掉50%的连接防止过拟合
model.add(Dense(num_classes, activation='softmax'))
# 最后的输出层，注意这里使用了softmax作为激活函数， 所以上面y_train = keras.utils.to_categorical(y_train, num_classes)
# 需要将标签做一个onehot编码

model.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy']) # 编译指定随时函数，优化器，指标(这里使用正确率)

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_test, y_test))
# 类似skearn训练数据，实际上keras和skearn类似，像水管，上面的步骤都是架设水管，水(数据)没放入水管，现在开始放水(训练数据)
# validation_data=(x_test, y_test)这里使用指定的测试集进行，如果不设置侧可以使用validation_split进行交叉验证

score = model.evaluate(x_test, y_test, verbose=0)  # 在测试集上测试数据verbose=0 表示简化输出模式
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# 下面是画出网络结构需要安装graphviz和pydot
from keras.utils import plot_model
plot_model(model, to_file='model2.png', show_layer_names=True, show_shapes=True)
