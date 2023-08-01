import torchvision.transforms as tfs
import os
from PIL import Image
from torch.utils import data
import numpy as np

class Dataset(data.Dataset):
    def __init__(self,path_root="../data/BHdata/",mode="Train",client_name="region_1"):
        super(Dataset,self).__init__()
        self.path_root = os.path.join(path_root + mode,client_name)
        self.rs_images_dir = os.listdir(os.path.join(self.path_root, "image"))
        self.rs_images = [os.path.join(self.path_root, "image", img) for img in self.rs_images_dir]
        

        self.gt_images_dir = os.listdir(os.path.join(self.path_root,"label"))
        self.gt_images = [os.path.join(self.path_root,"label",img) for img in self.rs_images_dir]
        
        if mode=="Train":
            self.rs_images = self.rs_images
            self.gt_images = self.gt_images

    def __getitem__(self, item):
        
        img = Image.open(self.rs_images[item])

        label = Image.open(self.gt_images[item])
        label = np.array(label)[:,:,0]
        img = tfs.ToTensor()(img)
        label = tfs.ToTensor()(label)

        return img, label

    def __len__(self):
        return len(self.rs_images)

# trainloader=data.DataLoader(
#             Dataset(path_root="./data/BHdata/",mode="Train",client_name="region_1"),
#             batch_size=4,shuffle=True,num_workers=4,pin_memory=True)
# print(trainloader)