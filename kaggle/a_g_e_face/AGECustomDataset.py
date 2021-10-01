import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from glob import glob
from PIL import Image
import os

class CustomAGEDataset(Dataset):
    def __init__(self, root_dir: str, transform: transforms = None):
        self.root = root_dir
        self.transform = transform
        self.ethnicityMap = {"white": 0, "black": 1, "asian": 2, "indian": 3, "others": 4}
        self.genderMap = {"male": 0, "female": 1}
        self.dataset = self.load_dataset(self.root)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        imgpath, age, eth, gender = self.dataset[idx]
        img = Image.open(imgpath).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = {'age': int(age), 'ethnicity': self.ethnicityMap[eth], 'gender': self.genderMap[gender]}

        return img, label

    def load_dataset(self, rootDir):
        dataset = []
        imgList = glob(os.path.join(rootDir, '*.jpg'))

        for idx, images in enumerate(imgList):
            basename = os.path.basename(images)
            splitname = basename.split("_")
            age = splitname[2]
            eth = splitname[0]
            gender = splitname[1]

            dataset.append((images, age, eth, gender))
            
        return dataset


def getDataloader(opt):
    mean = torch.tensor([0.5], dtype=torch.float32)
    std = torch.tensor([0.5], dtype=torch.float32)

    train_tranforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    trainset = CustomAGEDataset(root_dir=opt.train_dir, transform=train_tranforms)
    validationset = CustomAGEDataset(root_dir=opt.val_dir, transform=test_transforms)
    testset = CustomAGEDataset(root_dir=opt.test_dir, transform=test_transforms)

    trainIter = DataLoader(
        dataset=trainset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers
    )

    validIter = DataLoader(
        dataset=validationset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers
    )

    testIter = DataLoader(
        dataset=testset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers
    )

    return {'train': trainIter, 'validation': validIter, 'test': testIter}




if __name__ == '__main__':
    dir = r'C:\Users\mfbob\OneDrive\Desktop\age_gender_eth\test'
    MyCustomDataset = CustomAGEDataset(dir)
    print(len(MyCustomDataset))
    print(MyCustomDataset[0])
