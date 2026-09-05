from model import *
from datasets import *
from transforms import *


def model_factory(model_params):
    # ["mit_b0", "mit_b3", "unet_resnet34", "deeplab_v3plus"]
    
    model_name = model_params['name']
    in_channels = model_params['in_channels']
    proj_dim = model_params['proj_dim']
    num_classes = model_params['num_classes']
    
    if model_name == 'mit_b0':
        model = contrastive_mit_b0(
            in_channels=in_channels, projection_dim=proj_dim, num_classes=num_classes)
        return model
    
    elif model_name == 'mit_b3':
        model = contrastive_mit_b5(
            in_channels=in_channels, projection_dim=proj_dim, num_classes=num_classes)
        return model

    elif model_name == 'unet_resnet34':
        model = contrastive_Unet(
            in_channels=in_channels, projection_dim=proj_dim, num_classes=num_classes
        )
        return model
    
    elif model_name == 'deeplab_v3plus':
        print("deeplab_v3plus created")
        
    else:
        print(f"Model with the name {model_name} has not implemented.")


def dataset_factory(dataset_params):
     # ["Sen1Floods11", "Cityscapes", "Landsat8_Cloud", "Pascal_VOC2012"]
    
    dataset_name = dataset_params['name']
    
    if dataset_name == 'Sen1Floods11':

        # TODO: calculate mean & std for each dataset. Those are BEN-14K mean & std
        mean = np.array([627.1802, 676.7937, 428.5101, 1091.0465,  -17.2513,  -11.1336])
        std =  np.array([419.1856, 272.5272, 229.5443, 385.9456,   3.7207,   3.5236])
        
        transform = Sen1Floods11_transform(mean, std)
        dataset = Sen1Floods11_DS(
            '/kaggle/input/datasets/robertomarinoformica/sen1floods11-dataset/HandLabeled/S1Hand',
            '/kaggle/input/datasets/robertomarinoformica/sen1floods11-dataset/HandLabeled/S2Hand',
            '/kaggle/input/datasets/robertomarinoformica/sen1floods11-dataset/HandLabeled/LabelHand',
            transform=transform
        )
        
        return dataset
    
    elif dataset_name == 'Cityscapes':
        print("Cityscapes created")

    elif dataset_name == 'Landsat8_Cloud':
        print("Landsat8_Cloud created")

    elif dataset_name == 'Pascal_VOC2012':
        print("Pascal_VOC2012 created")
        
    else:
        print(f"Dataset with the name {dataset_name} has not implemented.")
