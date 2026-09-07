import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class contrastive_mit_b0(torch.nn.Module):
    def __init__(self, in_channels: int = 6, projection_dim: int = 128, num_classes: int = 2):
        super().__init__()
        
        # 1. MixTransformer-B0
        self.encoder = smp.encoders.get_encoder(
            name="mit_b0",
            in_channels=in_channels,
            depth=5,
            weights=None
        )
        
        # 2. projection per each piramyd layer (SegFormer-style)
        self.proj_layers = torch.nn.ModuleDict({
            '32':  torch.nn.Conv2d(32, 128, kernel_size=1),
            '64':  torch.nn.Conv2d(64, 128, kernel_size=1),
            '160': torch.nn.Conv2d(160, 128, kernel_size=1),
            '256': torch.nn.Conv2d(256, 128, kernel_size=1)
        })
        
        # 3. fuse after concatenation
        self.fuse_block = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )
        
        # 4. projection head
        self.contrast_head = torch.nn.Conv2d(256, projection_dim, kernel_size=1)
        
        # 5. seg head
        self.segmentation_head = torch.nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        
        
        B, _, H, W = x.size()
        
        # extracting features
        features = self.encoder(x)
        fused_features = []
        
        for f in features:
            channels = f.size(1)
            ch_key = str(channels)
            
            if ch_key in self.proj_layers:
                proj = self.proj_layers[ch_key](f)
                
                if proj.size(2) != H or proj.size(3) != W:
                    proj = F.interpolate(proj, size=(H, W), mode='bilinear', align_corners=False)
                    
                fused_features.append(proj)
                
        # concatenation [B, 512, H, W]
        x_fused = torch.cat(fused_features, dim=1)
        
        # features after fuse block [B, 256, H, W]
        x_features = self.fuse_block(x_fused)
        
        
        # contrastive learning
        z_spatial = self.contrast_head(x_features) # [B, projection_dim, H, W]
        z_flat = z_spatial.flatten(2).transpose(1, 2) # [B, H*W, projection_dim]
        
        # seg head
        logits = self.segmentation_head(x_features) # [B, num_classes, H, W]
        
        
        return (z_flat, logits)


class contrastive_mit_b5(torch.nn.Module):
    def __init__(self, in_channels: int = 6, projection_dim: int = 128, num_classes: int = 2):
        super().__init__()
        
        # 1. MixTransformer-B0
        self.encoder = smp.encoders.get_encoder(
            name="mit_b5",
            in_channels=in_channels,
            depth=5,
            weights=None
        )
        
        # 2. projection per each piramyd layer (SegFormer-style)
        self.proj_layers = torch.nn.ModuleDict({
            '64':  torch.nn.Conv2d(64, 128, kernel_size=1),
            '128':  torch.nn.Conv2d(128, 128, kernel_size=1),
            '320': torch.nn.Conv2d(320, 128, kernel_size=1),
            '512': torch.nn.Conv2d(512, 128, kernel_size=1)
        })
        
        # 3. fuse after concatenation
        self.fuse_block = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )
        
        # 4. projection head
        self.contrast_head = torch.nn.Conv2d(256, projection_dim, kernel_size=1)
        
        # 5. seg head
        self.segmentation_head = torch.nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        
        
        B, _, H, W = x.size()
        
        # extracting features
        features = self.encoder(x)
        fused_features = []
        
        for f in features:
            channels = f.size(1)
            ch_key = str(channels)
            
            if ch_key in self.proj_layers:
                proj = self.proj_layers[ch_key](f)
                
                if proj.size(2) != H or proj.size(3) != W:
                    proj = F.interpolate(proj, size=(H, W), mode='bilinear', align_corners=False)
                    
                fused_features.append(proj)
                
        # concatenation [B, 512, H, W]
        x_fused = torch.cat(fused_features, dim=1)
        
        # features after fuse block [B, 256, H, W]
        x_features = self.fuse_block(x_fused)
        
        
        # contrastive learning
        z_spatial = self.contrast_head(x_features) # [B, projection_dim, H, W]
        z_flat = z_spatial.flatten(2).transpose(1, 2) # [B, H*W, projection_dim]
        
        # seg head
        logits = self.segmentation_head(x_features) # [B, num_classes, H, W]
        
        
        return (z_flat, logits)


class contrastive_Unet(torch.nn.Module):
    def __init__(self, in_channels=6, projection_dim=128, num_classes=2):

        super().__init__()
        
        full_unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=in_channels, classes=num_classes)

        self.encoder = full_unet.encoder
        self.decoder = full_unet.decoder
        
        self.projection_head = torch.nn.Conv2d(16, projection_dim, kernel_size=1)
        self.segmentation_head = torch.nn.Conv2d(16, num_classes, kernel_size=1)


    def forward(self, x):

        features = self.encoder(x)
        features = self.decoder(features)

        print(features.shape)

        z_spatial = self.projection_head(features)
        z_flat = z_spatial.flatten(2).transpose(1, 2)

        logits = self.segmentation_head(features)


        return (z_flat, logits)


class contrastive_Unet_EfficientNet(torch.nn.Module):
    def __init__(self, in_channels=6, num_classes=2, proj_dim=128):

        super().__init__()
        
        full_unet = smp.Unet(
            encoder_name="efficientnet-b7",
            encoder_weights=None,
            in_channels=in_channels, classes=num_classes)

        self.encoder = full_unet.encoder
        self.decoder = full_unet.decoder
        
        self.projection_head = torch.nn.Conv2d(16, proj_dim, kernel_size=1)
        self.segmentation_head = torch.nn.Conv2d(16, num_classes, kernel_size=1)


    def forward(self, x):

        features = self.encoder(x)
        features = self.decoder(features)

        print(features.shape)
        z_spatial = self.projection_head(features)
        z_flat = z_spatial.flatten(2).transpose(1, 2)

        logits = self.segmentation_head(features)


        return (z_flat, logits)
    
