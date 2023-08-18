import torch
import os
from scipy import io
import random


class dataset():
    """
    加载数据，并进行预处理后保存
    """
    def __init__(self,root='', train=True, transform=None,target_transform=None):
        """
        :param root: 数据存放位置 默认存放在当前路径下
        :param train: 是否加载的是
        :param transform: 经过的变换
        """
        if train:
            self.data_file = "train_data.pt"
        else:
            self.data_file = "val_data.pt" #val_data
        self.root = root
        # self.train = train
        self.transform = transform
        self.target_transform = target_transform
        # 现在检查是否存在原始文件


        # 若原始文件存在，检查是否有 .pt文件
        if not (os.path.exists(os.path.join(self.root,self.data_file))):
            generate_train_data()

        self.data = torch.load(os.path.join(self.root, self.data_file))
        aaa = 1
        # if train:  # 如果加载训练集
        #     self.train_data =

    def __len__(self):
        """
        获得数据集 样本数量
        :return:  数量
        """
        # if self.train:
        #print(len(self.data))
        return len(self.data)

    def __getitem__(self, index):
        """
        获得数据集得中得一个数据
        :param index:  坐标
        :return:
        """
        # if self.train:
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode="RGB")
        # target = Image.fromarray(target.numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img


def get_data_random(data,num):
    im_size = 33
    h = data.shape[0]
    l = data.shape[1]
    inputs = []
    labels = []
    for n in range(num):
        x = random.randint(0,h-im_size)
        y = random.randint(0,l-im_size)
        sub_im = data[x:x+im_size,y:y+im_size]
        inputs.append(sub_im)
        labels.append(sub_im)
    return inputs,labels


def generate_train_data(path="dataset",data_type="NBA"):
    """
    该函数就是负责生成训练集,验证集
    :param path:
    :return:
    """
    if data_type !="all":
        all_path = os.path.join(path,data_type)
        trains = []
        tests = []
        n = 0

        dataFiles = []
        for roots,dirs,files in os.walk(all_path):
            if roots==all_path:
                for file in files:
                    if file.split(".")[-1]=="mat":
                        dataFiles.append(file)
        random.shuffle(dataFiles)
        trainFiles = dataFiles[0:1000]  # the training set
        testFiles = dataFiles[1000:]  # the test set

        for file in trainFiles:
            data = io.loadmat(os.path.join(all_path, file))['data']  # 这就是数据
            # data = data.transpose([2,0,1])
            trains.append(data)
            print(n+1)
            n+=1
        for file in testFiles:
            data = io.loadmat(os.path.join(all_path, file))['data']  # 这就是数据
            # data = data.transpose([2,0,1])
            tests.append(data)
            print(n + 1)
            n += 1
        train_inputs = torch.Tensor(trains)
        test_inputs = torch.Tensor(tests)
        training_set = (train_inputs)
        test_set = (test_inputs)

        with open(os.path.join("training_data", data_type, "train_data.pt"), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join("training_data", data_type, "val_data.pt"), 'wb') as f:
            torch.save(test_set, f)
        # with open(os.path.join(self.root, self.dir_name, self.testing_file), 'wb') as f:
        #     torch.save(test_set, f)
        print("文件已经打包为 .pt 文件")
    else:
        paths = ["NBA", "vehicle", "park"]
        trains = []
        tests = []
        n = 0
        for sub_path in paths:
            all_path = os.path.join(path, sub_path)


            dataFiles = []
            for roots, dirs, files in os.walk(all_path):
                if roots == all_path:
                    for file in files:
                        if file.split(".")[-1] == "mat":
                            dataFiles.append(file)
            random.shuffle(dataFiles)
            trainFiles = dataFiles[0:1000]  # the training set
            testFiles = dataFiles[1000:]  # the test set

            for file in trainFiles:
                data = io.loadmat(os.path.join(all_path, file))['data']  # 这就是数据
                # data = data.transpose([2,0,1])
                trains.append(data)
                print(n + 1)
                n += 1
            for file in testFiles:
                data = io.loadmat(os.path.join(all_path, file))['data']  # 这就是数据
                # data = data.transpose([2,0,1])
                tests.append(data)
                print(n + 1)
                n += 1
        train_inputs = torch.Tensor(trains)
        test_inputs = torch.Tensor(tests)
        training_set = (train_inputs)
        test_set = (test_inputs)

        with open(os.path.join("training_data", data_type, "train_data.pt"), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join("training_data", data_type, "val_data.pt"), 'wb') as f:
            torch.save(test_set, f)
        # with open(os.path.join(self.root, self.dir_name, self.testing_file), 'wb') as f:
        #     torch.save(test_set, f)
        print("文件已经打包为 .pt 文件")

if __name__ == "__main__":
    generate_train_data(path="original_data",data_type="all")