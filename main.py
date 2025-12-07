import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage import io, color, transform
import matplotlib.pyplot as plt

TRAIN_IMAGE_FOLDER = 'train_images'
OUTPUT_FOLDER = 'outputs'
MODEL_SAVE_PATH = 'colorization_model.pth'
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
IMAGE_SIZE = (224, 224)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ColorizationDataset(Dataset):
    def __init__(self, root_dir, size=IMAGE_SIZE):
        self.root_dir = root_dir
        self.size = size
        self.files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(self.files) == 0:
            raise RuntimeError(f"No photos in the folder '{root_dir}'! Put some colorful pictures there.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])

        rgb_img = io.imread(img_path)

        if len(rgb_img.shape) == 2:
            rgb_img = color.gray2rgb(rgb_img)
        if rgb_img.shape[2] == 4:
            rgb_img = rgb_img[:, :, :3]

        rgb_img = transform.resize(rgb_img, self.size)

        lab_img = color.rgb2lab(rgb_img)

        l_channel = (lab_img[:, :, 0] - 50.0) / 50.0

        ab_channels = lab_img[:, :, 1:] / 128.0

        l_tensor = torch.from_numpy(l_channel).unsqueeze(0).float()
        ab_tensor = torch.from_numpy(ab_channels.transpose((2, 0, 1))).float()

        return l_tensor, ab_tensor


class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, stride=2), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, padding=1, stride=2), nn.ReLU(), nn.BatchNorm2d(256),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, padding=1, stride=2, output_padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 3, padding=1, stride=2, output_padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, padding=1, stride=2, output_padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 2, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

def tensor_to_rgb(l_t, ab_t):
    if l_t.dim() == 4: l_t = l_t.squeeze(0)
    if ab_t.dim() == 4: ab_t = ab_t.squeeze(0)

    l_np = (l_t.cpu().detach().numpy().squeeze() * 50.0) + 50.0
    ab_np = ab_t.cpu().detach().numpy().transpose((1, 2, 0)) * 128.0

    lab_img = np.zeros((l_np.shape[0], l_np.shape[1], 3))
    lab_img[:, :, 0] = l_np
    lab_img[:, :, 1:] = ab_np

    try:
        rgb_img = color.lab2rgb(lab_img)
    except Exception as e:
        print(f"Conversion error: {e}")
        rgb_img = np.zeros_like(lab_img)

    return rgb_img


def save_sample_image(model, dataset, epoch):
    model.eval()
    with torch.no_grad():
        l, ab = dataset[0]
        l_input = l.unsqueeze(0).to(DEVICE)

        ab_pred = model(l_input)

        img_gray = tensor_to_rgb(l, torch.zeros_like(ab))  # Tylko L
        img_pred = tensor_to_rgb(l, ab_pred.squeeze(0))  # L + przewidziane ab
        img_real = tensor_to_rgb(l, ab)  # L + prawdziwe ab

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img_gray);
        axs[0].set_title("Input (Grayscale)")
        axs[1].imshow(img_pred);
        axs[1].set_title("AI Output (Colorized)")
        axs[2].imshow(img_real);
        axs[2].set_title("Ground Truth (Original)")

        for ax in axs: ax.axis('off')

        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        plt.savefig(f"{OUTPUT_FOLDER}/epoch_{epoch + 1}.png")
        plt.close()
    model.train()

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print(f"Start program. Device: {DEVICE}")
    print("Data preparation...")

    try:
        dataset = ColorizationDataset(TRAIN_IMAGE_FOLDER)
    except RuntimeError as e:
        print(e)
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Loaded {len(dataset)} photos.")

    model = ColorizationNet().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0

        for i, (l_input, ab_target) in enumerate(dataloader):
            l_input, ab_target = l_input.to(DEVICE), ab_target.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(l_input)
            loss = criterion(outputs, ab_target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Loss: {avg_loss:.5f}")

        save_sample_image(model, dataset, epoch)

    print(f"TTraining completed in time: {(time.time() - start_time) / 60:.2f} min.")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved as {MODEL_SAVE_PATH}")
    print(f"Check the folder '{OUTPUT_FOLDER}', to see the results!")


if __name__ == '__main__':
    main()