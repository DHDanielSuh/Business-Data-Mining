import cv2
import pandas as pd
import os
import argparse
from jinja2 import Environment, FileSystemLoader
import json

__author__ = "Suh, Dong Hyun"
__copyright__ = "Copyright 2019, The AI Vehicle Recognition Project"
__credits__ = ["Suh, Dong Hyun"]
__license__ = "GPL"
__version__ = "0.1.0"
__maintainer__ = ["Suh, Dong Hyun"]
__email__ = ["dan90511@mindslab.ai"]



file_path = "/home/daniel/Downloads/dataset_0711"


class Writer:
    def __init__(self, path, width, height, depth, database='Unknown', segmented=0):
        file_loader = FileSystemLoader('template')
        environment = Environment(loader=file_loader)
        self.annotation_template = environment.get_template('annotation.xml')

        abspath = os.path.abspath(path)

        self.template_parameters = {
            'path': abspath,
            'filename': os.path.basename(abspath),
            'folder': os.path.basename(os.path.dirname(abspath)),
            'width': width,
            'height': height,
            'depth': depth,
            'database': database,
            'segmented': segmented,
            'objects': []
        }

    def addObject(self, name, xmin, ymin, xmax, ymax, pose='Unspecified', truncated=0, difficult=0):
        self.template_parameters['objects'].append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'pose': pose,
            'truncated': truncated,
            'difficult': difficult,
        })

    def save(self, annotation_path):
        with open(annotation_path, 'w') as file:
            content = self.annotation_template.render(**self.template_parameters)
            file.write(content)


class GETINFO:
    def __init__(self, file_path):
        self.file_path = file_path


    def EAST(self, file_path):
        self.file_path = file_path
        filenames = os.listdir(file_path)

        for filename in filenames:
            full_filename = os.path.join(file_path, filename)
            ext = os.path.splitext(full_filename)[-1]
            if ext == '.jpg':
                img = cv2.imread(full_filename, cv2.IMREAD_UNCHANGED)
                height = img.shape[0]
                width = img.shape[1]
                depth = img.shape[2]

                writer = Writer(full_filename, width, height, depth)
                filename_wo_ext = os.path.splitext(filename)[0]
                txt_filename = filename_wo_ext + '.txt'
                df = pd.read_csv(os.path.join(file_path, txt_filename), header=None,
                            names=['x_min', 'y_min', 'x_max', 'y_max', 'answer'],
                            usecols=(0, 1, 4, 5, 8))
                for index, row in df.iterrows():
                    writer.addObject(row['answer'], row['x_min'], row['y_min'], row['x_max'], row['y_max'])
                writer.save(os.path.join(file_path, filename_wo_ext) + '.xml')

    def YOLO(self, file_path):
        self.file_path = file_path
        filenames = os.listdir(file_path)

        for filename in filenames:
            full_filename = os.path.join(file_path, filename)
            ext = os.path.splitext(full_filename)[-1]
            if ext == '.jpg':
                img = cv2.imread(full_filename, cv2.IMREAD_UNCHANGED)
                height = img.shape[0]
                width = img.shape[1]
                depth = img.shape[2]

                writer = Writer(full_filename, width, height, depth)
                filename_wo_ext = os.path.splitext(filename)[0]
                txt_filename = filename_wo_ext + '.txt'
                df = pd.read_csv(os.path.join(file_path, txt_filename), header=None,
                                 names=['answer','center_x', 'center_y', 'box_width', 'box_height'],
                                 sep='\s+')
                for index, row in df.iterrows():
                    writer.addObject(int(row['answer']), int(round(row['center_x']*width-(row['box_width']/2)*width)),
                                     int(round(row['center_y']*height-(row['box_height']/2)*height)),
                                     int(round(row['center_x']*width+(row['box_width']/2)*width)),
                                     int(round(row['center_y']*height+(row['box_height']/2)*height)))
                writer.save(os.path.join(file_path, filename_wo_ext) + '.xml')


    def FACENET(self, file_path):
        self.file_path = file_path
        filenames = os.listdir(file_path)

        for filename in filenames:
            full_filename = os.path.join(file_path, filename)
            ext = os.path.splitext(full_filename)[-1]
            if ext == '.jpg':
                img = cv2.imread(full_filename, cv2.IMREAD_UNCHANGED)
                height = img.shape[0]
                width = img.shape[1]
                depth = img.shape[2]

                writer = Writer(full_filename, width, height, depth)
                filename_wo_ext = os.path.splitext(filename)[0]
                json_filename = filename_wo_ext + '.json'

                data = open(os.path.join(file_path, json_filename)).read()
                result = json.loads(data)

                x_min = result['vertices'][0][0]
                y_min = result['vertices'][0][1]
                x_max = result['vertices'][3][0]
                y_max = result['vertices'][3][1]

                dict = {'x_min': [x_min], 'y_min': [y_min], 'x_max': [x_max], 'y_max': [y_max]}
                df = pd.DataFrame.from_dict(dict)

                for index, row in df.iterrows():
                    writer.addObject('bill', row['x_min'], row['y_min'], row['x_max'], row['y_max'])
                writer.save(os.path.join(file_path, filename_wo_ext) + '.xml')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', type=str, default='EAST',
                        choices=['EAST', 'YOLO', 'FACENET'],
                        help='Please Choose Operation Mode among (EAST, YOLO, FACENET)')
    args = parser.parse_args()
    op = args.op
    voc = GETINFO(file_path)
    if op == 'EAST':
        voc.EAST(file_path)
    elif op == 'YOLO':
        voc.YOLO(file_path)
    elif op == 'FACENET':
        voc.FACENET(file_path)

if __name__=="__main__":
    main()


# def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
#     """
#     Call in a loop to create terminal progress bar
#     @params:
#         iteration   - Required  : current iteration (Int)
#         total       - Required  : total iterations (Int)
#         prefix      - Optional  : prefix string (Str)
#         suffix      - Optional  : suffix string (Str)
#         decimals    - Optional  : positive number of decimals in percent complete (Int)
#         length      - Optional  : character length of bar (Int)
#         fill        - Optional  : bar fill character (Str)
#     """
#     percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
#     filledLength = int(length * iteration // total)
#     bar = fill * filledLength + '-' * (length - filledLength)
#     print('\r%s|%s| %s%% (%s/%s)  %s' % (prefix, bar, percent, iteration, total, suffix), end = '\r')
#     # Print New Line on Complete
#     if iteration == total:
#         print("\n")



import cv2
import pandas as pd
import os
import argparse
from jinja2 import Environment, FileSystemLoader
import json

__author__ = "Suh, Dong Hyun"
__copyright__ = "Copyright 2019, The AI Vehicle Recognition Project"
__credits__ = ["Suh, Dong Hyun"]
__license__ = "GPL"
__version__ = "0.1.0"
__maintainer__ = ["Suh, Dong Hyun"]
__email__ = ["dan90511@mindslab.ai"]



file_path = "/home/daniel/Downloads/xmltest"


class Writer:
    def __init__(self, path, width, height, depth):
        file_loader = FileSystemLoader('template')
        environment = Environment(loader=file_loader)
        self.annotation_template = environment.get_template('annotation.xml')

        abspath = os.path.abspath(path)

        self.template_parameters = {
            'path': abspath,
            'filename': os.path.basename(abspath),
            'folder': os.path.basename(os.path.dirname(abspath)),
            'width': width,
            'height': height,
            'depth': depth,
            'objects': []
        }

    def addObject(self, result, name='plate', groundtruth=' '):
        self.template_parameters['objects'].append({
            'name': name,
            'result': result,
            'groundtruth': groundtruth,
        })

    def save(self, annotation_path):
        with open(annotation_path, 'w') as file:
            content = self.annotation_template.render(**self.template_parameters)
            file.write(content)



def TO_XML(file_path, result):
    filenames = os.listdir(file_path)

    for filename in filenames:
        full_filename = os.path.join(file_path, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == '.jpg':
            img = cv2.imread(full_filename, cv2.IMREAD_UNCHANGED)
            height = img.shape[0]
            width = img.shape[1]
            depth = img.shape[2]

            writer = Writer(full_filename, width, height, depth)
            filename_wo_ext = os.path.splitext(filename)[0]
            writer.addObject(result)
            writer.save(os.path.join(file_path, filename_wo_ext) + '.xml')



if __name__=="__main__":
    result = '서울 12345'
    TO_XML(file_path, result)
