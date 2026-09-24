from datasets import Cityscapes_DS, Sen1Floods11_DS
from utils import get_class_weights


def main():
    Cityscape_ds = Cityscapes_DS("/kaggle/input/datasets/electraawais/cityscape-dataset")
    SenFloods_ds = Sen1Floods11_DS(
        '/kaggle/input/datasets/robertomarinoformica/sen1floods11-dataset/HandLabeled/S1Hand',
        '/kaggle/input/datasets/robertomarinoformica/sen1floods11-dataset/HandLabeled/S2Hand',
        '/kaggle/input/datasets/robertomarinoformica/sen1floods11-dataset/HandLabeled/LabelHand',
    )
    
    get_class_weights(Cityscape_ds, 19)
    get_class_weights(SenFloods_ds, 2)

if __name__ == "__main__":
    main()
