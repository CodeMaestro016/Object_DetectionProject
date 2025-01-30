import cv2
import numpy as np
import os

# Set the base path for model files
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Files"))
weights_path = os.path.join(base_path, "yolov4.weights")
config_path = os.path.join(base_path, "yolov4.cfg")
coco_names_path = os.path.join(base_path, "coco.names")

# Image file path
image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Images/car.jpg"))

# Check if all required files exist
for path in [weights_path, config_path, coco_names_path, image_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)

# Load COCO class names
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the image
image = cv2.imread(image_path)
height, width, _ = image.shape

# Convert image to blob and perform forward pass
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(output_layers)

# Process the outputs
boxes, confidences, class_ids = [], [], []
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression (NMS)
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Check if any objects were detected before proceeding
if len(indices) > 0:
    for i in indices.flatten():  # Convert indices to a list
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # Green bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
else:
    print("⚠ No objects detected in the image.")

# Show Image
cv2.imshow("YOLO Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
