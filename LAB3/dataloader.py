import pandas as pd
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms


def getData(mode):
    if mode == "train":
        df = pd.read_csv("./train.csv")
        path = df["Path"].tolist()
        label = df["label"].tolist()
        return path, label

    elif mode == "valid":
        df = pd.read_csv("./valid.csv")
        path = df["Path"].tolist()
        label = df["label"].tolist()
        return path, label

    else:
        df = pd.read_csv("./resnet_152_test.csv")
        path = df["Path"].tolist()
        return path


class LeukemiaLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        if mode == "test":
            self.img_name = getData(mode)
        else:
            self.img_name, self.label = getData(mode)
            print("> Found %d images..." % (len(self.img_name)))
        self.mode = mode

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """

        train_transform = transforms.Compose(
            [
                transforms.RandomRotation(degrees=20),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # step 1
        path = self.root + self.img_name[index]
        # step 2
        if self.mode == "test":
            image = Image.open(path).convert("RGB")
            img = test_transform(image)
            return img
        else:
            label = self.label[index]

        # step 3
        image = Image.open(path).convert("RGB")
        if self.mode == "train":
            img = train_transform(image)
        else:
            img = test_transform(image)

        return img, label
