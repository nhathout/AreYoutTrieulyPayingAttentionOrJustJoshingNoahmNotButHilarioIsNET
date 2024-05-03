def load_model(model_path):
    # Load the model architecture
    model = models.resnet50(pretrained=False)
    # Load the pre-trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image_path):
    # Load an image and transform it
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict(model, image):
    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
    return predicted

def main():
    model_path = '/projectnb/dl523/students/jbardwic/face_model.pth'
    image_path = '/projectnb/dl523/students/jbardwic/CourtVideo.mp4'
    
    # Load the model
    model = load_model(model_path)
    
    # Preprocess the image
    image = preprocess_image(image_path)
    
    # Predict the class
    prediction = predict(model, image)
    print("Predicted Class:", prediction.item())

if __name__ == '__main__':
    main()
