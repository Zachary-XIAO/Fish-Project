# transfer all JSON files in labelme folder into a single JSON file,
# which will be used later in the training model.

import json
import PIL
import glob
import numpy as np
from labelme import utils
from shapely.geometry import Polygon


class DataPrepare(object):

    def __init__(self, labelme_json=[], save_json_path='./new.json'):
        """
        Define some global parameters.

        :param labelme_json: List of all labelme json file path
        :param save_json_path: json save path
        """
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = [{'supercategorr': 'FishHead', 'id': 1, 'name': 'FishHead'},
                           {'supercategorr': 'FishBody', 'id': 2, 'name': 'FishBody'},
                           {'supercategorr': 'Fish', 'id': 3, 'name': 'Fish'}
                           ]
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, 'r') as fp:
                data = json.load(fp)  # load json files
                self.images.append(self.image(data, num))
                for shapes in data['shapes']:
                    label = shapes['label']
                    points = shapes['points']
                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1
        print(self.categories)

    def image(self, data, num):
        image = {}
        img = utils.img_b64_to_arr(data['imageData'])  # parse image data
        height, width = img.shape[:2]
        img = None
        image['height'] = height
        image['width'] = width
        image['id'] = num + 1
        image['file_name'] = data['imagePath'].split('/')[-1]

        self.height = height
        self.width = width

        return image

    def annotation(self, points, label, num):
        """
        We need to turn the polygons into coordinates.
        polygons are the mask we drew in labelme.
        The mask in COCO format is a series of coordinates surrounding the object.
        Together with getcatid(), getbbox(), mask2box(), polygons_to_mask

        :param points:
        :param label:
        :param num:
        :return:
        """
        annotation = {}
        annotation['segmentation'] = [list(np.asarray(points).flatten())]
        poly = Polygon(points)
        area_ = round(poly.area, 6)
        annotation['area'] = area_
        annotation['iscrowd'] = 0
        annotation['image_id'] = num + 1
        annotation['bbox'] = list(map(float, self.getbbox(points)))
        annotation['category_id'] = self.getcatid(label)
        annotation['id'] = self.annID
        return annotation

    def getcatid(self, label):
        for category in self.categories:
            if label == category['name']:
                return category['id']
        return -1

    def getbbox(self, points):
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        """
        calculate the bounding box by its mask.
        mask: [h, w] a picture of 0,1
        1 is the object, only needs to calculate the upper left and bottom right
        """
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        cols = index[:, 1]
        # get the left top point
        left_top_r = np.min(rows)
        left_top_c = np.min(cols)
        # get the right bottom point
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(cols)

        return [left_top_c, left_top_r, right_bottom_c - left_top_c, right_bottom_r - left_top_r]

    def polygons_to_mask(self, img_shape, polygons):
        # turn the polygons into coordinates

        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        """
        There are several fields in the COCO format.
        So, we need to prepare each of them.
        """
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)


if __name__ == '__main__':
    # csv to coco
    labelme_json = glob.glob('./../labelme/dataset/*.json')  # get the json files we drew in labelme
    print(labelme_json)

    DataPrepare(labelme_json, './trainval.json')  # transfer all the json files into one single file named trainval.json
