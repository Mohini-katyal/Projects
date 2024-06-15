import cv2
import torch
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
import heapq
import time

def find_brightest_areas(depth_map, window_size=20):
    max_intensity = np.max(depth_map)
    global max_intensity_coords

    max_intensity_coords=np.column_stack(np.where(depth_map == max_intensity))
    for i in range(max_intensity-5 , max_intensity):
        new_coords = np.column_stack(np.where(depth_map == i))
        max_intensity_coords = np.append(max_intensity_coords, new_coords, axis=0)

    brightest_areas = []
    for coords in max_intensity_coords:
        top_left_x = max(0, coords[1] - window_size // 2)
        top_left_y = max(0, coords[0] - window_size // 2)
        bottom_right_x = min(depth_map.shape[1], top_left_x + window_size)
        bottom_right_y = min(depth_map.shape[0], top_left_y + window_size)

        brightest_area = depth_map[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        brightest_areas.append(((top_left_x, top_left_y, bottom_right_x, bottom_right_y), brightest_area))

    return brightest_areas

def show_image(depth_map):
    depth_map_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_map), cv2.COLORMAP_BONE)
    #cv2.imshow("Depth Map", depth_map_colored)
    cv2.waitKey(1)

def find_box_for_point(depth_map, point):
    h, w,ch = depth_map.shape
    boxes = [
        ((0, 0), (h//3, w//3)),
        ((0, w//3), (h//3, 2*w//3)),
        ((0, 2*w//3), (h//3, w)),
        ((h//3, 0), (2*h//3,w//3)),
        ((h//3, w//3), (2*h//3, 2*w//3)),
        ((h//3, 2*w//3), (2*h//3, w)),
        ((2*h//3, 0), (h, w//3)),
        ((2*h//3, w//3), (h, 2*w//3)),
        ((2*h//3, 2*w//3), (h, w)),
    ]

    for i, box in enumerate(boxes):
        if box[0][0] <= point[0] <= box[1][0] and box[0][1] <= point[1] <= box[1][1]:
            return i + 1

    return None

class ModelType(Enum):
    DPT_LARGE = "DPT_Large"     
    DPT_Hybrid = "DPT_Hybrid"   
    MIDAS_SMALL = "MiDaS_small" 

class Midas():
    def __init__(self, modelType: ModelType=ModelType.DPT_LARGE):
        self.midas = torch.hub.load("isl-org/MiDaS", modelType.value)
        self.modelType = modelType

    def useCUDA(self):
        if torch.cuda.is_available():
            print('Using CUDA')
            self.device = torch.device("cuda")
        else:
            print('Using CPU')
            self.device = torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()

    def transform(self):
        print('Transform')
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.modelType.value == "DPT_Large" or self.modelType.value == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def predict(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depthMap = prediction.cpu().numpy()
        depthMap = cv2.normalize(depthMap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_map_gray = cv2.applyColorMap(depthMap, cv2.COLORMAP_BONE)
        show_image(depth_map_gray)

        global max_intensity_coords
        brightest_areas = find_brightest_areas(depth_map_gray)
        rows, cols,ch = depth_map_gray.shape

        frequency=[0,0,0,0,0,0,0,0,0]
        for i in max_intensity_coords:
            point=i
            box_number = find_box_for_point(depth_map_gray,point)
            for j in range(len(frequency)):
                if j==box_number-1:
                    frequency[j]+=1

        matrix = np.zeros((3, 3))
        largest_indices = heapq.nlargest(2, range(len(frequency)), key=frequency.__getitem__)

        max_index = largest_indices[0]
        secondmax_index = largest_indices[1]

        count = 0
        for i in range(3):
            for j in range(3):
                if count == max_index or count == secondmax_index:
                    matrix[i][j] = 1
                count += 1

        print(matrix)
        return depthMap

    def livePredict(self, camera_index=0):
        print('Starting webcam (press q to quit)...')
        capObj = cv2.VideoCapture(camera_index)
        while True:
            ret, frame = capObj.read()
            if not ret:
                print("Failed to retrieve frame")
                break
            depthMap = self.predict(frame)

            # Convert depthMap to 3 channels to match frame
            depthMap_colored = cv2.cvtColor(depthMap, cv2.COLOR_GRAY2BGR)

            # Horizontally stack frame and depthMap
            combined = np.hstack((frame, depthMap_colored))
            cv2.imshow('Combined', combined)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

            time.sleep(1)  # Delay for one second

        capObj.release()
        cv2.destroyAllWindows()

def run(modelType: ModelType):
    midasObj = Midas(modelType)
    midasObj.useCUDA()
    midasObj.transform()
    midasObj.livePredict()

if __name__ == '__main__':
    while True:
        run(ModelType.MIDAS_SMALL)
