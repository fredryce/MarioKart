import os, glob
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator
from tf_pose import common
import cv2
import pandas as pd
from itertools import chain
import datetime
import numpy as np


def find_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou


class OpenPose(TfPoseEstimator):

	def __init__(self, file, target_size = (432,368), convert_csv = True):
		self.e = TfPoseEstimator(file, target_size= target_size)
		self.w, self.h = model_wh('432x368')
		self.convert_csv = convert_csv
		self.df, self.temp_list = self.prepare_csv()

	def detect(self, frame):
		humans = self.e.inference(frame, resize_to_default=(self.w > 0 and self.h > 0), upsample_size=4.0)
		return humans

	def draw_humans(self, npimg, humans, imgcopy=False):
	    if imgcopy:
	        npimg = np.copy(npimg)
	    image_h, image_w = npimg.shape[:2]
	    center = None
	    for human in humans:
		    if 0 in human.body_parts.keys() and 14 in human.body_parts.keys() and 15 in human.body_parts.keys():
		        body_part = human.body_parts[0]
		        body_part_reye = human.body_parts[14]
		        body_part_leye = human.body_parts[15]
		        #center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
		        center = np.array([[int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5)],[int(body_part_reye.x * image_w + 0.5), int(body_part_reye.y * image_h + 0.5)],[int(body_part_leye.x * image_w + 0.5), int(body_part_leye.y * image_h + 0.5)]], dtype=np.float32)
		    break
	    return center

	def top_bottom_ratio(self, top_distance, bottom_distance ,body, center):
		if body[1] < center[1]:
			return ((center[1] - body[1])/top_distance)*100
		return ((center[1] - body[1])/bottom_distance)*100


	def side_ratio(self, slope, intercept, body, center):
		x_value = (body[1] - intercept)/slope
		distance_center_to_x = center[0] - x_value
		distance_to_body = body[0] - center[0]
		return (distance_to_body/distance_center_to_x) *100


	def find_distance(self, bed_coord, center):
		topLeft, topRight, bottomLeft, bottomRight = bed_coord
		slope = (bottomLeft[1] - topLeft[1])/(bottomLeft[0] - topLeft[0])
		intercept = bottomLeft[1] - (slope*bottomLeft[0])
		top_distance = center[1] - topLeft[1]
		bottom_distance = bottomLeft[1] - center[1]
		assert top_distance > 0 and bottom_distance > 0, "Top and bottom distance not greater than 0"
		return (slope, intercept, top_distance, bottom_distance)



	def find_centers(self, bed_coord):
		topLeft, topRight, bottomLeft, bottomRight = bed_coord
		slope1 = (bottomRight[1] - topLeft[1])/(bottomRight[0] - topLeft[0])
		slope2 = (bottomLeft[1]-topRight[1])/(bottomLeft[0]-topRight[0])
		intercept1 = bottomRight[1] - (slope1*bottomRight[0])
		intercept2 = bottomLeft[1] - (slope2*bottomLeft[0])
		x_coord = (intercept1- intercept2) / (slope2 - slope1)
		y_coord = (slope1 * x_coord) + intercept1
		return (x_coord, y_coord)
		
	def prepare_csv(self):
		if not self.convert_csv:
			return None, None
		exist_csv = glob.glob(os.path.join('./', r'*.csv'))
		if exist_csv:
			return pd.read_csv(exist_csv[0], index_col=0), list()
		head = list(chain.from_iterable((common.CocoPart(x).name+'X', common.CocoPart(x).name+'Y') for x in range(common.CocoPart.Background.value)))
		head.append('Label')
		head.insert(0, 'Time')
		return pd.DataFrame(columns=head), list()

