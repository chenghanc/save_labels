# save_labels

1. Save labels on a list of new images using trained yolo models with this project [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)

2. Python API code is modified according to this project [gengyanlei/fire-smoke-detect-yolov4](https://github.com/gengyanlei/fire-smoke-detect-yolov4)

3. `libdarknet.so` might be need to recompile on your PC

## Dependencies

- opencv-python

## OS

- Ubuntu 18.04 LTS
- macOS

## Demo: Call the API function

```python
import os
from os import getcwd
from save_labels_api import Detect

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
```

---

## References

1. [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
2. [gengyanlei/fire-smoke-detect-yolov4](https://github.com/gengyanlei/fire-smoke-detect-yolov4)