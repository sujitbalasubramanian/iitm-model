import cv2
import time
import torch
from sort import *
from pathlib import Path
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import LOGGER, check_img_size, increment_path, non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_sync
from binary_classifier import BinaryClassifier
from flask import Flask, Response

app = Flask(__name__)

def save_snapshot(img, unique_id, frame_number):
    cv2.imwrite(f"snapshot/snapshot_{unique_id}_{frame_number}.jpg", img)
    print(f"Snapshot captured for ID {unique_id} at frame {frame_number}")

def is_helmet(model, frame, x1, y1, x2, y2):
    cropped_img = frame[y1:y2, x1:x2]
    cropped_img = cv2.resize(cropped_img, (128, 128))  
    cropped_img = torch.from_numpy(cropped_img).permute(2, 0, 1).unsqueeze(0).float()
    output = model(cropped_img)
    helmet_present = output.item() > 0.5
    if helmet_present:
        return "Helmet: False"
    else:
        return "Helmet: True"

frame = 0
unique_ids = {}
min_appearances = 1

def draw_boxes(img, bbox, identities=None, categories=None, names=None, roi_vertices=None, model=None, frame=frame):
    offset=(0, 0)
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        cat_name = names[int(cat)]
        label = f"{id}:{cat_name}"

        if(cat_name == "motorcycle"):
            condition = is_helmet(model, img, x1, y1, x2, y2)
            label = f"{label} {condition}"
            if id not in unique_ids:
                unique_ids[id] = 0
            unique_ids[id] += 1

            if unique_ids[id] >= min_appearances and unique_ids[id] == 1:
                save_snapshot(img[y1:y2, x1:x2], id, frame)

        if roi_vertices is None or (
            cv2.pointPolygonTest(roi_vertices, (x1, y1), False) >= 0 and
            cv2.pointPolygonTest(roi_vertices, (x2, y1), False) >= 0 and
            cv2.pointPolygonTest(roi_vertices, (x2, y2), False) >= 0 and
            cv2.pointPolygonTest(roi_vertices, (x1, y2), False) >= 0
        ):

            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2),(255,191,0), 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,191,0), -1)
            cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
            [255, 255, 255], 1)
            cv2.circle(img, data, 3, (255,191,0),-1)

        if roi_vertices is not None:
            cv2.polylines(img, [roi_vertices], isClosed=True, color=(0, 255, 0), thickness=2)
 
    return img

im0
    
def detect():
    #............Initialization............#
    weights = 'weights/best_last.pt'
    source = 'videos/D303erer_20240216183826.mp4'
    helmet_weights = 'weights/best_model.pth'
    device = 0
    imgsz = (640, 640)
    conf_thres = 0.25
    iou_thres = 0.45
    half = False
    visualize = False
    augment = False
    view_img = False
    exist_ok = False
    save_txt = False
    name = 'exp'
    webcam = False
    project = 'runs/detect'
    global im0

    #roi_vertices = np.array([[275, 382], [870,382], [1245, 704], [43, 704]], np.int32)
    #roi_vertices = np.array([[287,877], [812,777], [4000,1224], [1200, 6000]], np.int32)

    #..............SORT.................#
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                    min_hits=sort_min_hits,
                    iou_threshold=sort_iou_thresh) 

    #..........Model.................#
    device = select_device(device)
    print(device)
    half &= device.type != 'cpu' 
    model = DetectMultiBackend(weights, device=device, dnn=False, data='custom_dataset.yaml')
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)
    print('Yolov5 Load Sucess')

    helmet_model = BinaryClassifier()
    helmet_model.load_state_dict(torch.load(helmet_weights))


    #...........Inference...........#
    if not webcam:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True) 
    t0 = time.time()
    dt, seen = [0.0, 0.0, 0.0], 0

    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)
        dt[2] += time_sync() - t3

        for i, det in enumerate(pred):
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            
            p = Path(p)
            s += '%gx%g ' % im.shape[2:]

            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                #..................USE TRACK FUNCTION....................
                #pass an empty array to sort
                dets_to_sort = np.empty((0,6))
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                              np.array([x1, y1, x2, y2, 
                                                        conf, detclass])))
                
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)

        
                # draw boxes for visualization
                if len(tracked_dets)>0:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    draw_boxes(im0, bbox_xyxy, identities, categories, names, model=helmet_model,frame=frame)
                
                frame+=1

            if view_img:
                im0 = cv2.resize(im0,(1280,720))
                cv2.imshow(str(p), im0)
                cv2.waitKey(1) 

def stream():
    try:
        ret, buffer = cv2.imencode('.jpg', im0)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as err:
        print(err)

@app.route('/video_feed')
def webcam_display():
    return Response(detect(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    detect()
    app.run(debug=True)

