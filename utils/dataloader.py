import torchvision.datasets as dset
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import parameters as p

############ Perform Transformation ############
transform_Img = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

train_dataset = dset.ImageFolder(root=p.TRAIN_DIR, transform=transform_Img)
train_data_loader = DataLoader(train_dataset, batch_size=p.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)

test_dataset = dset.ImageFolder(root=p.TEST_DIR, transform=transform_Img)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=0)
