
# SECL
Official repository for training and inference with Supervised Energy Contrastive Loss (SECL), developed as part of my PhD research.


**Abstract**


We propose a novel supervised pixel-wise contrastive loss function, termed Supervised Energy Contrastive Loss (SECL). Operating within the energy-based model (EBM) paradigm, SECL integrates a variety of mathematical methods from different computer vision tasks and adapts them to the domain of semantic segmentation. We provide a conceptual explanation and a formal mathematical description of the proposed loss function. The effectiveness of this approach is validated through extensive experiments using both lightweight and deep convolutional neural networks (CNNs), as well as vision transformers (ViTs), on the Sen1Floods11 and Cityscapes datasets. We demonstrate that SECL outperforms or matches the traditional Supervised Contrastive Loss (SupCon) in down-stream metrics while significantly reducing computational overhead, as supported by our provided efficiency benchmarks. Furthermore, we visualize the loss landscape to demonstrate the formation of deep energy valleys. These visualizations prove that the behavior of SECL strictly aligns with the core philosophy of EBMs.
