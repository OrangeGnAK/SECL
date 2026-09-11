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
        model = contrastive_Unet_EfficientNet(
            in_channels=in_channels, projection_dim=proj_dim, num_classes=num_classes
        )
        return model
    
    elif model_name == 'unet_efficientnet_b7':
        model = contrastive_Unet_(
            in_channels=in_channels, projection_dim=proj_dim, num_classes=num_classes
        )
        return model
        
    else:
        print(f"Model with the name {model_name} has not implemented.")


def dataset_factory(dataset_params):
     # ["Sen1Floods11", "Cityscapes"]
    
    dataset_name = dataset_params['name']
    
    if dataset_name == 'Sen1Floods11':

        # SenFloods mean & std!!!
        mean = np.array([1195.1572, 1344.4296, 1379.89, 2578.4109, -10.4345, -17.308])
        std =  np.array([860.6037, 731.6119, 734.5892, 1029.2466, 4.1806, 4.8804])
        
        transform = Sen1Floods11_transform(mean, std)
        dataset = Sen1Floods11_DS(
            '/kaggle/input/datasets/robertomarinoformica/sen1floods11-dataset/HandLabeled/S1Hand',
            '/kaggle/input/datasets/robertomarinoformica/sen1floods11-dataset/HandLabeled/S2Hand',
            '/kaggle/input/datasets/robertomarinoformica/sen1floods11-dataset/HandLabeled/LabelHand',
            transform=transform
        )
        
        return dataset
    
    elif dataset_name == 'Cityscapes':

        mean = np.array([73.1584, 82.9089, 72.3924])
        std =  np.array([47.6758, 48.4942, 47.7365])

        # TODO: change splitting logic for Cityscape. I can split it here via constructor
        # and then in utils.py divide spliting and init logic in two separate functions
        # so i can call split only for the SenFloods
        transform = Cityscape_transform(mean, std)
        dataset = Cityscapes_DS(
            '/kaggle/input/datasets/electraawais/cityscape-dataset',
            mode='train',
            transform=transform)

        return dataset
    

    elif dataset_name == 'Landsat8_Cloud':
        print("Landsat8_Cloud created")

    elif dataset_name == 'Pascal_VOC2012':
        print("Pascal_VOC2012 created")
        
    else:
        print(f"Dataset with the name {dataset_name} has not implemented.")
