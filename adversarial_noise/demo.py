import torch
from noise import FGSM, PDG
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO


model = models.resnet18(pretrained=True)
model.eval()

# Load an image
url = "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_tensor = preprocess(img).unsqueeze(0)

# Move the model and tensor to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
img_tensor = img_tensor.to(device)

# Get the model's prediction on the original image
with torch.no_grad():
    output = model(img_tensor)
_, pred = output.max(1)
original_label = pred.item()

# Apply FGSM
fgsm = FGSM(model, targeted=False)
epsilon = 0.03
adv_img_fgsm = fgsm(img_tensor, original_label, epsilon)

# Apply PDG
pdg = PDG(model,  targeted=False)
k = 10
adv_img_pdg = pdg(img_tensor, original_label, k, epsilon)

# Get the model's predictions on the adversarial images
with torch.no_grad():
    output_fgsm = model(adv_img_fgsm)
    output_pdg = model(adv_img_pdg)

_, pred_fgsm = output_fgsm.max(1)
_, pred_pdg = output_pdg.max(1)

label_fgsm = pred_fgsm.item()
label_pdg = pred_pdg.item()

# Denormalize function
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# Plot the images
def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
imshow(img_tensor, title=f'Original Image\nLabel: {original_label}')

plt.subplot(1, 3, 2)
imshow(adv_img_fgsm, title=f'FGSM Image\nLabel: {label_fgsm}')

plt.subplot(1, 3, 3)
imshow(adv_img_pdg, title=f'PDG Image\nLabel: {label_pdg}')

plt.show()