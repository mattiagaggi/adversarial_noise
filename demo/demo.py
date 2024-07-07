from demo_utils import download_image, imshow
import torch
from adversarial_noise import FGSM, PDG
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

# Mapping of ImageNet class indices to human-readable labels
imagenet_classes = {i: line.strip() for i, line in enumerate(open("imagenet-classes.txt"))}

# Load a pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)
model.eval()

# URL of the image to be downloaded and analyzed
url = "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg"

# Download and convert the image to RGB format
img = download_image(url).convert('RGB')

# Preprocess the image: resize, center crop, convert to tensor, and normalize
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply the preprocessing steps to the image and add a batch dimension
img_tensor = preprocess(img).unsqueeze(0)

# Determine the appropriate device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
img_tensor = img_tensor.to(device)

# Get the model's prediction on the original image
with torch.no_grad():
    output = model(img_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)

# Extract the predicted class and confidence for the original image
_, pred = output.max(1)
original_label = pred.item()
original_confidence = probabilities[0, original_label].item()
print(f"Original label: {original_label} ({imagenet_classes[original_label]}) with confidence {original_confidence:.4f}")

# Apply Fast Gradient Sign Method (FGSM) attack
fgsm = FGSM(model, targeted=False)
epsilon = 0.03
adv_img_fgsm = fgsm(img_tensor, original_label, epsilon)
print("FGSM attack applied")

# Apply Projected Gradient Descent (PDG) attack
pdg = PDG(model, targeted=False)
k = 10
adv_img_pdg = pdg(img_tensor, original_label, k, epsilon)
print("PDG attack applied")

# Get the model's predictions on the adversarial images
with torch.no_grad():
    output_fgsm = model(adv_img_fgsm)
    probabilities_fgsm = torch.nn.functional.softmax(output_fgsm, dim=1)
    output_pdg = model(adv_img_pdg)
    probabilities_pdg = torch.nn.functional.softmax(output_pdg, dim=1)

# Extract the predicted class and confidence for the adversarial images
_, pred_fgsm = output_fgsm.max(1)
_, pred_pdg = output_pdg.max(1)

label_fgsm = pred_fgsm.item()
label_pdg = pred_pdg.item()

confidence_fgsm = probabilities_fgsm[0, label_fgsm].item()
confidence_pdg = probabilities_pdg[0, label_pdg].item()

print(f"FGSM label: {label_fgsm} ({imagenet_classes[label_fgsm]}) with confidence {confidence_fgsm:.4f}")
print(f"PDG label: {label_pdg} ({imagenet_classes[label_pdg]}) with confidence {confidence_pdg:.4f}")

# Display the original and adversarial images with their respective labels and confidences
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
imshow(img_tensor, title=f'Original Image\nLabel: {original_label} ({imagenet_classes[original_label]})\nConfidence: {original_confidence:.4f}')

plt.subplot(1, 3, 2)
imshow(adv_img_fgsm, title=f'FGSM Image\nLabel: {label_fgsm} ({imagenet_classes[label_fgsm]})\nConfidence: {confidence_fgsm:.4f}')

plt.subplot(1, 3, 3)
imshow(adv_img_pdg, title=f'PDG Image\nLabel: {label_pdg} ({imagenet_classes[label_pdg]})\nConfidence: {confidence_pdg:.4f}')

# Save the plot with the displayed images
plt.savefig('output.png')
plt.show()
