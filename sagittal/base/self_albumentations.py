import albumentations as A
import cv2
import os

# Define the augmentation pipeline
transform = A.Compose([
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
    A.ToGray(p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5)
])

def read_yolo_labels(label_path):
    """Read YOLO format labels from a file."""
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def write_yolo_labels(label_path, labels):
    """Write YOLO format labels to a file."""
    with open(label_path, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")

def main(image_path, label_path, output_image_dir, output_label_dir, augmented_num):
    # Read image
    image = cv2.imread(image_path)
    # Read labels
    labels = read_yolo_labels(label_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    for i in range(augmented_num):
        # Apply data augmentation
        transformed = transform(image=image)
        transformed_image = transformed['image']
        
        output_image_path = os.path.join(output_image_dir, f"{base_name}_{i}.jpg")
        output_label_path = os.path.join(output_label_dir, f"{base_name}_{i}.txt")

        # Save augmented image and labels
        cv2.imwrite(output_image_path, transformed_image)
        write_yolo_labels(output_label_path, labels)

if __name__ == "__main__":
    # Define directories
    input_image_dir = '/home/under1/Detect/jeongui/sagittal/base/images/train'
    input_label_dir = '/home/under1/Detect/jeongui/sagittal/base/labels/train'
    output_image_dir = './augmented/images/train'
    output_label_dir = './augmented/labels/train'
    
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    for image_name in os.listdir(input_image_dir):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            base_name = os.path.splitext(image_name)[0]
            image_path = os.path.join(input_image_dir, image_name)
            label_path = os.path.join(input_label_dir, base_name + '.txt')
            
            if os.path.exists(label_path):
                main(image_path, label_path, output_image_dir, output_label_dir, augmented_num=5)