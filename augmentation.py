import os
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

base_dir = "data/test"
target_count = 1200
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

augment_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(3),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.3)
])

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def augment_image(img_path, save_path):
    image = Image.open(img_path).convert('L')
    augmented = augment_transforms(image)
    augmented.save(save_path)

for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    image_files = [f for f in os.listdir(class_path) if is_image_file(f)]
    current_len = len(image_files)

    if current_len > target_count:
        print(f"[↓] Downsampling {class_name} from {current_len} to {target_count}")
        random.shuffle(image_files)
        keep = image_files[:target_count]
        remove = image_files[target_count:]
        for file in remove:
            os.remove(os.path.join(class_path, file))

    elif current_len < target_count:
        to_generate = target_count - current_len
        print(f"[↑] Augmenting {to_generate} images for {class_name}")
        for i in tqdm(range(to_generate)):
            original = random.choice(image_files)
            orig_path = os.path.join(class_path, original)
            save_path = os.path.join(class_path, f"aug_{i}.jpg")
            augment_image(orig_path, save_path)

print("\nAll test classes now contain exactly 1200 samples.")

base_dir = "data/train"
target_count = 6000
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

augment_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(3),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.3)
])

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def augment_image(img_path, save_path):
    image = Image.open(img_path).convert('L')
    augmented = augment_transforms(image)
    augmented.save(save_path)

for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    image_files = [f for f in os.listdir(class_path) if is_image_file(f)]
    current_len = len(image_files)

    if current_len > target_count:
        print(f"[↓] Downsampling {class_name} from {current_len} to {target_count}")
        random.shuffle(image_files)
        keep = set(image_files[:target_count])
        for file in image_files:
            if file not in keep:
                os.remove(os.path.join(class_path, file))

    elif current_len < target_count:
        to_generate = target_count - current_len
        print(f"[↑] Augmenting {to_generate} images for {class_name}")
        for i in tqdm(range(to_generate)):
            original = random.choice(image_files)
            orig_path = os.path.join(class_path, original)
            save_path = os.path.join(class_path, f"aug_{i}.jpg")
            augment_image(orig_path, save_path)

    else:
        print(f"[✓] {class_name} already has {current_len} images")

print("\nAll classes are now balanced to exactly 6000 images.")