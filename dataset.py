import os
import numpy as np
import torch
import random
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
import pandas as pd
import math
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self,
                 data_path: str, 
                 split: str):
        data = pd.read_csv(data_path)
        configs = []
        for i in data.index:
            configs.append(data.loc[i].values[1:6])
        random.seed(4321)
        random.shuffle(configs)
        self.configs = configs[:int(len(configs) * 0.75)] if split == "train" else configs[int(len(configs) * 0.75):]

    def __len__(self):
        return len(self.configs)
    
    def __getitem__(self, idx):
        # normalization
        hw, cin, cout, ks, s = self.configs[idx]
        data = torch.Tensor([
            (math.log(hw, math.e) + 0.1) / math.log(240, math.e), # hw
            math.log(cin, math.e) / math.log(3480, math.e), # cin
            math.log(cout, math.e) / math.log(3480, math.e), # cout
            (ks - 1) / 6, # ks
            s - 1
        ])
        return data, 0.0


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    
    

class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
#       =========================  OxfordPets Dataset  =========================
            
#         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                               transforms.CenterCrop(self.patch_size),
# #                                               transforms.Resize(self.patch_size),
#                                               transforms.ToTensor(),
#                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
#         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                             transforms.CenterCrop(self.patch_size),
# #                                             transforms.Resize(self.patch_size),
#                                             transforms.ToTensor(),
#                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#         self.train_dataset = OxfordPets(
#             self.data_dir,
#             split='train',
#             transform=train_transforms,
#         )
        
#         self.val_dataset = OxfordPets(
#             self.data_dir,
#             split='val',
#             transform=val_transforms,
#         )
        
#       =========================  CelebA Dataset  =========================
    
        # train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                       transforms.CenterCrop(148),
        #                                       transforms.Resize(self.patch_size),
        #                                       transforms.ToTensor(),])
        
        # val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                     transforms.CenterCrop(148),
        #                                     transforms.Resize(self.patch_size),
        #                                     transforms.ToTensor(),])
        
        # self.train_dataset = MyCelebA(
        #     self.data_dir,
        #     split='train',
        #     transform=train_transforms,
        #     download=False,
        # )
        
        # # Replace CelebA with your dataset
        # self.val_dataset = MyCelebA(
        #     self.data_dir,
        #     split='test',
        #     transform=val_transforms,
        #     download=False,
        # )

#       ===========================  MyDataset  ====================================
        

        self.train_dataset = MyDataset(
            self.data_dir,
            split='train'
        )
        
        # Replace CelebA with your dataset
        self.val_dataset = MyDataset(
            self.data_dir,
            split='test'
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

def make_divisible(v):
    if v == 3: return v
    new_v = max(8, int(v + 8 / 2) // 8 * 8)
    if new_v < 0.9 * v:
        new_v += 8
    return new_v

def data_validation(data, cdata):
    newlist = []
    for da in cdata:
        value = abs(da - data)
        newlist.append(value)

    # newlist = list(np.asarray(newlist).T)    
    cda = list(newlist).index(min(newlist))
    redata = cdata[cda]
    return redata


hw_candidate = [
    1, 2, 3, 5, 6, 7, 10, 11, 12, 13, 14, 20, 21,
    22, 23, 24, 25, 26, 27, 28, 40, 42, 44, 46, 48, 50,
    52, 54, 56, 80, 84, 88, 92, 96, 100, 104, 108, 112, 160,
    176, 192, 208, 224
]
channel_candidate = [
    3, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 104, 112,
    120, 128, 144, 160, 168, 176, 184, 192, 200, 208, 216, 224, 240, 256,
    288, 312, 320, 336, 352, 360, 368, 384, 400, 416, 432, 448, 480, 504,
    512, 528, 552, 576, 600, 624, 640, 648, 672, 768, 832, 864, 896, 960,
    1024, 1152, 1248, 1280, 1344, 1408, 1440, 1472, 1536, 1600, 1664, 1728, 1792, 1920,
    2016, 2048, 2112, 2208, 2304, 2400, 2496, 2560, 2592, 2688, 2816, 2944, 3072, 3200,
    3328, 3456
]

def process_single_data(item, refer_min_max = None):
    ''' refer_min_max is always None by now
    '''
    hw, cin, cout, ks, s = item
    if refer_min_max != None:
        hw = (hw - refer_min_max[0][0]) / (refer_min_max[0][1] - refer_min_max[0][0])
        hw = min(hw, 1)
        hw = max(hw, 0)
        
        cin = (cin - refer_min_max[1][0]) / (refer_min_max[1][1] - refer_min_max[1][0])
        cin = min(cin, 1)
        cin = max(cin, 0)
        
        cout = (cout - refer_min_max[2][0]) / (refer_min_max[2][1] - refer_min_max[2][0])
        cout = min(cout, 1)
        cout = max(cout, 0)
        
    hw = data_validation(math.exp(hw * math.log(240, math.e) - 0.1), hw_candidate)
    cin = data_validation(int(math.exp(cin * math.log(3480, math.e))), channel_candidate)
    cout = data_validation(int(math.exp(cout * math.log(3480, math.e))), channel_candidate)
    ks = data_validation(ks * 6 + 1, [1, 3, 5, 7])
    s = data_validation(s + 1, [1, 2])
    return[hw, cin, cout, ks, s]

def post_process(data, orig_data=None, refer_data=None):
    '''
    data: data value for post process
    orig_data: data value in case to print original data for data reconstruction. If orig_data is not None, post_process() will return (data, orig_data)
    refer_data: help add a normalization in testing phase. refer_data is the data distribution of training data, help scale the data range from range 0-1 to range (0.01 quantile - 0.99 quantile).
        refer_data is always None by now.
    '''
    import numpy as np
    if refer_data != None:
        # import pdb; pdb.set_trace()
        min_max = [item for item in zip(np.quantile(refer_data, 0.01, axis=0)[:3], np.quantile(refer_data, 0.99, axis=0)[:3])]
    else:
        min_max = None

    post_data = []
    if orig_data != None:
        for item, orig_item in zip(data, orig_data):
            post_data.append([process_single_data(item, min_max), process_single_data(orig_item, min_max)])
    else:
        for item in data:
            post_data.append(process_single_data(item, min_max))

    return post_data