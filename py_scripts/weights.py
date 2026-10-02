from datasets import Cityscapes_DS, Sen1Floods11_DS
from utils import get_class_weights

"""
Cityscape
tensor([6.1053, 1.0482, 3.7859, 0.3298, 0.3289, 0.2023, 0.0609, 0.0954, 2.6672,
        0.3390, 0.7273, 0.2536, 0.0644, 1.2027, 0.3622, 0.4132, 0.7968, 0.0946,
        0.1224], device='cuda:0')

SenFloods
tensor([1.7644, 0.2356], device='cuda:0')
"""
def main():
    Cityscape_ds = Cityscapes_DS("/kaggle/input/datasets/electraawais/cityscape-dataset")
    SenFloods_ds = Sen1Floods11_DS(
        '/kaggle/input/datasets/robertomarinoformica/sen1floods11-dataset/HandLabeled/S1Hand',
        '/kaggle/input/datasets/robertomarinoformica/sen1floods11-dataset/HandLabeled/S2Hand',
        '/kaggle/input/datasets/robertomarinoformica/sen1floods11-dataset/HandLabeled/LabelHand',
    )
    
    print(get_class_weights(Cityscape_ds, 19))
    print(get_class_weights(SenFloods_ds, 2))

if __name__ == "__main__":
    main()
