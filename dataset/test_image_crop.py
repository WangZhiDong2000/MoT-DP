from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import glob

# Find first jpg image
jpg_files = glob.glob('/home/wang/Project/carla_garage/data/**/rgb/*.jpg', recursive=True)
if not jpg_files:
    print("No jpg files found")
    exit(1)

img_path = jpg_files[0]
print(f"Testing with image: {img_path}")

# Load original image
img = Image.open(img_path).convert('RGB')
print(f"Original size: {img.size}")  # (width, height)

# Apply transform
image_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.crop((0, 0, 1024, 384))),  # Crop top 384 pixels
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img_tensor = image_transform(img)
print(f"Tensor shape: {img_tensor.shape}")
print(f"Tensor range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")

# Denormalize for visualization
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
img_denorm = img_tensor * std + mean
img_denorm = torch.clamp(img_denorm, 0, 1)

# Convert to numpy for visualization
img_vis = np.moveaxis(img_denorm.numpy(), 0, -1)
img_vis = (img_vis * 255).astype(np.uint8)
print(f"Visualization shape: {img_vis.shape}")
print(f"Visualization range: [{img_vis.min()}, {img_vis.max()}]")

# Save visualization
plt.figure(figsize=(20, 7.5))
plt.imshow(img_vis)
plt.title(f'Cropped Image (1024x384)')
plt.axis('off')
plt.tight_layout()
plt.savefig('/tmp/test_crop_vis.png', dpi=150, bbox_inches='tight')
print("Saved visualization to /tmp/test_crop_vis.png")
