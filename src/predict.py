import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from src.model import get_model
from src.data_preprocessing import PlantVillageDataset

num_classes=len(PlantVillageDataset('with-augmentation').classes)

def prepare_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def visualize_prediction(image, probabilities, classes):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    top_k = 5
    top_probs, top_classes = torch.topk(probabilities, top_k)
    plt.barh(range(top_k), top_probs.cpu().numpy())
    plt.yticks(range(top_k), [classes[i] for i in top_classes])
    plt.title(f'Top {top_k} Class Probabilities')
    plt.xlabel('Probability')
    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    plt.close()

def predict(image_path, model_path, device='cuda'):
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    
    # Load the dataset to get the classes
    dataset = PlantVillageDataset('without-augmentation')
    classes = dataset.classes
    
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    image = Image.open(image_path).convert('RGB')
    input_tensor = prepare_image(image_path).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()

    visualize_prediction(image, probabilities, classes)
    return classes[predicted_class], probabilities[predicted_class].item()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict plant disease from an image.")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--model_path", default='models/saved_models/plant_disease_model.pth', help="Path to the trained model")
    parser.add_argument("--device", default='cuda', choices=['cuda', 'cpu'], help="Device to use for inference")
    args = parser.parse_args()

    predicted_class, confidence = predict(args.image_path, args.model_path, args.device)
    print(f"The leaf is predicted to be: {predicted_class} with {confidence:.2f} confidence")
    print("Prediction visualization saved as 'prediction_visualization.png'")