from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


# 命令行的初始化
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset", required=False, default="./dataset",
                help="path to the dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to the output plot")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
                help="path to the output model")
args = vars(ap.parse_args())

# 参数的设置
INIT_LR = 1e-4
EPOCH = 20
BS = 32

print("[INFO] loading images ...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]

    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    # 归一化
    image = preprocess_input(image)

    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)
# 进行one-hot编码
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# stratify=labels根据标签的比例进行切分
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

aug = ImageDataGenerator(
    rotation_range=20, # 旋转角
    zoom_range=0.15, # 缩放
    width_shift_range=0.2, # 宽方向平移
    height_shift_range=0.2, # 长方向平移
    shear_range=0.15, # 偏移参数
    horizontal_flip=True, # 翻转
    fill_mode="nearest" # 填充
)
print("[INFO] make model ...")
# 构建模型
baseModel = MobileNetV2(weights = "imagenet", include_top = False,
                        input_tensor = Input(shape=(224, 224, 3)))

headModel = baseModel.output
# 平均池化
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
# 拉平
headModel = Flatten(name="flatten")(headModel)
# 全连接层
headModel = Dense(128, activation="relu")(headModel)
# dropout
headModel = Dropout(0.5)(headModel)
# 最后的线性层输出
headModel = Dense(2, activation="softmax")(headModel)

# 定义模型
model = Model(inputs=baseModel.input, outputs=headModel)

# 冻结输出层
for layer in baseModel.layers:
    layer.trainable = False

# 进行模型的训练
print("[INFO] comping model ...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCH)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])
print("[INFO] training head ...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS), # 数据增强
    steps_per_epoch=len(trainX) // BS, # 训练的长度
    validation_data=(testX, testY), # 验证集
    validation_steps=len(testX) // BS, # 验证集训练的长度
    epochs=EPOCH
)
print("[INFO] evaluating network ...")
# 进行预测
predIdex = model.predict(testX, batch_size=BS)
predIdex = np.argmax(predIdex, axis=1)

print(classification_report(testY.argmax(axis=1), predIdex,
                            target_names=lb.classes_))
# 模型保存
print("[INFO] saving mask detector model ...")
model.save(args["model"], save_format="h5")

# 进行画图的操作
N = EPOCH
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history['loss'], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="validation_data")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

