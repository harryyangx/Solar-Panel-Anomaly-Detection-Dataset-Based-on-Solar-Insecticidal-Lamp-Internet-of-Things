import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('//runs/detect/11n_3000/weights/compare_hsppf.pt')
    model.val(data='/home/pc/solar_panels_train/datasets/datasets_3000/data.yaml',
              split='test',
              imgsz=640,
              batch=32,
              iou=0.6,
              rect=False,
              save_json=True,
              project='runs/test',
              name='11n_3000',
              plots=True,
              workers=16,
              save_txt=False,
              save_conf=False,
              save_crop=False
              )