import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2, os, glob, tqdm
from ultralytics import YOLO

train_img_path = './dataset/train/images'
train_label_path = './dataset/train/labels'

test_img_path = './dataset/test/images'
test_label_path = './dataset/test/labels'

transform = A.Compose([
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='yolo', min_area=1024, min_visibility=0.1, label_fields=['class_labels']))

def image_augmentation(image_path, bboxes, classes): 
    image = cv2.imread(image_path) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    augmented = transform(image=image, bboxes=bboxes, class_labels=classes) 
    augmented_image = augmented['image'] 
    augmented_bboxes = augmented['bboxes'] 
    augmented_classes = augmented['class_labels'] 
    folder, filename = os.path.split(image_path) 
    name, ext = os.path.splitext(filename) 
    new_filename = f"{name}_2{ext}" 
    new_image_path = os.path.join(train_img_path, new_filename) 
    cv2.imwrite(new_image_path, augmented_image) 
    new_label_filename = f"{name}_2.txt" 
    new_label_path = os.path.join(train_label_path, new_label_filename) 
    
    with open(new_label_path, 'w') as label_file: 
        for cls, bbox in zip(augmented_classes, augmented_bboxes): 
            label_file.write(f"{cls} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

def read_yolo_labels(label_path): 
    bboxes = [] 
    classes = [] 
    with open(label_path, 'r') as file: 
        lines = file.readlines() 
        for line in lines: 
            parts = line.strip().split() 
            cls = int(parts[0]) 
            bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])] 
            classes.append(cls) 
            bboxes.append(bbox) 
    return bboxes, classes

def process_images_in_directory(input_image_dir, input_label_dir): 
    for root, dirs, files in os.walk(input_image_dir): 
        for file in files: 
            if file.lower().endswith('.jpg'): 
                image_path = os.path.join(root, file) 
                label_path = os.path.join(input_label_dir, file.replace(os.path.splitext(file)[1], '.txt')) 
                bboxes, classes = read_yolo_labels(label_path) 
                image_augmentation(image_path, bboxes, classes)

process_images_in_directory(train_img_path, train_label_path)

model = YOLO('runs/detect/train16/weights/best.pt')


model = YOLO('mcr/yolov8n.pt')
model.train(data='dataset/data.yaml', epochs=5, imgsz=640, batch=16)
metrics = model.val()
print(metrics)

def output_result(result, image): 
    boxes = result.boxes.xyxy.cpu().numpy() 
    confidences = result.boxes.conf.cpu().numpy() 
    classes = result.boxes.cls.cpu().numpy() 
    for box, conf, cls in zip(boxes, confidences, classes): 
        x1, y1, x2, y2 = map(int, box) 
        label = f'{model.names[int(cls)]} {conf:.2f}' 
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) 
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Assuming the video file is named 'Volkswagen.mp4'
video_path = 'dataset/Volkswagen.mp4'
cap = cv2.VideoCapture(video_path)

# Prepare a VideoWriter to save the processed video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    output_result(results[0], frame)
    out.write(frame)

cap.release()
out.release()
