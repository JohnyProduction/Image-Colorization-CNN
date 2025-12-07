import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform
from main import ColorizationNet

MODEL_PATH = 'colorization_model.pth'
IMAGE_PATH = 'test.jpg'
SAVE_NAME = 'result.png'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def colorize_image(image_path, model_path):
    print("Loading the model...")
    model = ColorizationNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    print(f"Image processing: {image_path}")
    original_img = io.imread(image_path)

    if original_img.shape[-1] == 4:
        original_img = original_img[:, :, :3]

    if len(original_img.shape) == 3:
        original_img_gray = color.rgb2gray(original_img)
    else:
        original_img_gray = original_img

    img_resized = transform.resize(original_img_gray, (224, 224))

    img_rgb_mock = color.gray2rgb(img_resized)
    lab_img = color.rgb2lab(img_rgb_mock)

    l_channel = (lab_img[:, :, 0] - 50.0) / 50.0
    l_tensor = torch.from_numpy(l_channel).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        ab_output = model(l_tensor)

    ab_output = ab_output.cpu().squeeze(0).numpy().transpose((1, 2, 0)) * 128.0

    lab_output = np.zeros((224, 224, 3))
    lab_output[:, :, 0] = (l_channel * 50.0) + 50.0
    lab_output[:, :, 1:] = ab_output

    rgb_output = color.lab2rgb(lab_output)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img_gray, cmap='gray')
    plt.title("Original (B&W)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(rgb_output)
    plt.title("Colored by AI")
    plt.axis('off')

    plt.savefig(SAVE_NAME)
    print(f"The result was saved as: {SAVE_NAME}")
    plt.show()


if __name__ == "__main__":
    colorize_image(IMAGE_PATH, MODEL_PATH)