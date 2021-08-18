import os
from os import getcwd
import cv2
import numpy as np
import darknet
import time

class Detect:
    def __init__(self, metaPath, configPath, weightPath, gpu_id=2, batch=1):
        assert batch == 1, "batch==1"
        #darknet.set_gpu(gpu_id)
        network, class_names, class_colors = darknet.load_network(
            configPath,
            metaPath,
            weightPath,
            batch_size=batch
        )
        self.network = network
        self.class_names = class_names
        self.class_colors = class_colors

    def bbox2point(self, bbox):
        x, y, w, h = bbox
        xmin = x - (w / 2)
        xmax = x + (w / 2)
        ymin = y - (h / 2)
        ymax = y + (h / 2)
        return (xmin, ymin, xmax, ymax)

    def point2bbox(self, point):
        x1, y1, x2, y2 = point
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = (x2 - x1)
        h = (y2 - y1)
        return (x, y, w, h)

    def convert2relative(self, image, bbox):
        """
        YOLO format use relative coordinates for annotation
        """
        x, y, w, h = bbox
        height, width, _ = image.shape
        return x/width, y/height, w/width, h/height

    def save_annotations(self, name, image, detections, class_names):
        """
        Files saved with image_name.txt and relative coordinates
        """
        file_name = os.path.splitext(name)[0] + ".txt"
        with open(file_name, "w") as f:
            for label, confidence, bbox in detections:
                x, y, w, h = self.convert2relative(image, bbox)
                label = class_names.index(label)
                f.write("{} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h))

    def image_detection(self, image_path, network, class_names, class_colors, thresh):
        # Darknet doesn't accept numpy images.
        # Create one with image we reuse for each detect
        width = darknet.network_width(network)
        height = darknet.network_height(network)
        darknet_image = darknet.make_image(width, height, 3)

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
        darknet.free_image(darknet_image)
        image = darknet.draw_boxes(detections, image_resized, class_colors)
        cv2.putText(image, "Heads:" + str(len(detections)), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

    def predict_image(self, image_name, thresh=0.25, is_show=True, save_path=''):

        start = time.time()
        draw_bbox_image, detections = self.image_detection(image_name, self.network, self.class_names, self.class_colors,
                                                           thresh)
        end = time.time()
        run_time = "{:.3f} s".format(end-start)
        print("Done: {}".format(run_time))
        if is_show:
            darknet.print_detections(detections, coordinates=True)
            if save_path:
                self.save_annotations(image_name, draw_bbox_image, detections, self.class_names)
                cv2.imwrite(save_path, draw_bbox_image)
            return draw_bbox_image
        return detections


if __name__ == '__main__':

    detect = Detect(metaPath=r'custom.data',
                    configPath=r'custom.cfg',
                    weightPath=r'custom.weights',
                    gpu_id=1)

    # read folder
    # getting the current directory
    print(os.getcwd())
    image_root = r'custom'
    save_root = r'../output'
    # changing the current directory to one with images
    os.chdir(image_root)
    print(os.getcwd())
    wd = getcwd()
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for name in os.listdir(image_root):
        print("Enter Image Path: " + wd + name + ": Predicted in ")
        draw_bbox_image = detect.predict_image(os.path.join(image_root, name), save_path=os.path.join(save_root, name))
