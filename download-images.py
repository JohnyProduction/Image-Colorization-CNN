import os
import requests
import time

FOLDER_NAME = 'train_images'
NUM_IMAGES = 100
IMG_SIZE = 300


def download_images():
    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME)
        print(f"Folder created: {FOLDER_NAME}")

    print(f"Starting download {NUM_IMAGES} photos...")

    for i in range(NUM_IMAGES):
        try:
            url = f"https://picsum.photos/{IMG_SIZE}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                file_path = os.path.join(FOLDER_NAME, f"image_{i + 1}.jpg")
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"[{i + 1}/{NUM_IMAGES}] Downloaded: image_{i + 1}.jpg")
            else:
                print(f"[{i + 1}/{NUM_IMAGES}] Download error.")

            time.sleep(0.1)

        except Exception as e:
            print(f"An error occurred: {e}")

    print("\nDone! You can run it! main.py")


if __name__ == '__main__':
    download_images()