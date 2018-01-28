import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf

learning_rate=1e-4
train_iteration=2500
dropout=0.6
batch_size=50
validation_size=2000
image_to_display=10
#读取数据
def read_file(filename):
    data=pd.read_csv(filename)
    # print('data({0[0]},{0[1]})'.format(data.shape))
    # print(data.head())
    #获取到特征
    images=data.iloc[:,1:].values
    images=images.astype(np.float)
    images=np.multiply(images,1.0/225.0)
    # print('images({0[0]},{0[1]})'.format(images.shape))
    #获取图片大小
    image_size=images.shape[1]
    # print("image_size=>{0}".format(image_size))
    #获取图片宽度和高度
    image_width=image_height=np.ceil(np.sqrt(image_size)).astype(np.uint8)
    print("image_width=>{0}\nimage_height=>{1}".format(image_width,image_height))
    # 获取图片标签
    labels_flat = data.iloc[:, :1].values.ravel()
    print("labels_flat({0})".format(labels_flat))
    print("labels_flat[{0}]=>{1}".format(image_to_display, labels_flat[image_to_display]))
    labels_count = np.unique(labels_flat).shape[0]
    print("labels_count=>{0}".format(labels_count))
    return (images,image_size,image_width,image_height,labels_flat,labels_count)
#显示图片
def display(img,image_width,image_height):
    one_image=img.reshape(image_width,image_height)
    plt.axis('off')
    plt.imshow(one_image,cmap=cm.binary)
    plt.show()
# display(images[image_to_display])

#热编码
def dense_to_one_hot(labels_dense,num_classes):
    num_labels=labels_dense.shape[0]
    print(num_labels)
    index_offset=np.arange(num_labels)*num_classes
    labels_one_hot=np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()]=1
    return labels_one_hot


images,image_size,image_width,image_height,labels_flat,labels_count=read_file("kaggle_data/train.csv")
labels=dense_to_one_hot(labels_flat,labels_count).astype(np.uint8)
print("labels({0[0]},{0[1]})".format(labels.shape))
print("labels[{0}]=>{1}".format(image_to_display,labels[image_to_display]))

validation_images=images[:validation_size]
validation_labels=labels[:validation_size]
train_images=images[validation_size:]
train_labels=labels[validation_size:]
#初始化偏置参数
def weight_variable(shape):
    inital=tf.truncated_normal(shape,stddev=0.1)
    return  tf.Variable(inital)
def bias_variable(shape):
    inital=tf.constant(0.1,shape=shape)
    return tf.Variable(inital)
#卷积层
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

#池化层
def max_pool_2x2(x):
    return  tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x=tf.placeholder("float",shape=[None,image_size])
y_=tf.placeholder("float",shape=[None,labels_count])

W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
#(40000,784)=>(40000,28,28,1)
image=tf.reshape(x,[-1,image_width,image_height,1])

h_conv1=tf.nn.relu(conv2d(image,W_conv1)+b_conv1)
#print(h_conv1.get_shape())=>40000,28,28,32
h_pool1=max_pool_2x2(h_conv1)
#print(h_pool1.get_shape())=>40000,14,14,32

W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
#print(h_conv1.get_shape())=>40000,14,14,64
h_pool2=max_pool_2x2(h_conv2)
#print(h_pool1.get_shape())=>40000,7,7,64

W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

W_fc2=weight_variable([1024,labels_count])
b_fc2=bias_variable([labels_count])
y=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

cross_entropy=-tf.reduce_sum((y_*tf.log(y)))
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
correct_predction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_predction,tf.float32))
predict=tf.argmax(y,1)

epochs_completed=0
index_in_epoch=0
num_examples=train_images.shape[0]

def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    start=index_in_epoch
    index_in_epoch+=batch_size
    if index_in_epoch>num_examples:
        epochs_completed+=1
        perm=np.arange(num_examples)
        np.random.shuffle(perm)
        train_images=train_images[perm]
        train_labels=train_labels[perm]
        start=0
        index_in_epoch=batch_size
        assert  batch_size<=num_examples
    end=index_in_epoch
    return train_images[start:end],train_labels[start:end]

init=tf.global_variables_initializer()
sess=tf.InteractiveSession()
sess.run(init)

train_acccuracies=[]
validation_accuracies=[]
x_range=[]
display_step=1
for i in range(train_iteration):
    xs,ys=next_batch(batch_size)
    if i%display_step==0 or (i+1)==train_iteration :
        train_accuracy=accuracy.eval(feed_dict={x:xs,y_:ys,keep_prob:1.0})
        if(validation_size):
            validation_accuracy=accuracy.eval(feed_dict={x:validation_images,y_:validation_labels,keep_prob:1.0})
            print("training_accuracy/validation_accuracy=>%.2f/%.2f for step %d"%(train_accuracy,validation_accuracy,i))
            validation_accuracies.append(validation_accuracy)
        else:
            print("training_accuracy=>%.4f for step %d"%(train_accuracy,i))
        train_acccuracies.append(train_accuracy)
        x_range.append(i)
        if i%(display_step*10)==0 and i:
            display_step*=10
    sess.run(train_step,feed_dict={x:xs,y_:ys,keep_prob:dropout})
if(validation_size):
        validation_accuracy=sess.run(accuracy,feed_dict={x:validation_images,y_:validation_labels,keep_prob:1.0})
        print("validation_accuracy=>%.4f"%validation_accuracy)
        plt.plot(x_range,train_acccuracies,'-b',label='Training')
        plt.plot(x_range,validation_accuracies,'-g',label='Validation')
        plt.legend(loc="lower right",frameon=False)
        plt.ylim(ymax=1.1,ymin=0.7)
        plt.xlabel('step')
        plt.show()
