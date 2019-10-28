import cv2
import os
from jinja2 import Environment, FileSystemLoader

file_path = "/home/xmltest"

def TO_XML(file_path, result, name= 'plate', groundtruth = None):
    filenames = os.listdir(file_path)

    for filename in filenames:
        full_filename = os.path.join(file_path, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == '.jpg':
            img = cv2.imread(full_filename, cv2.IMREAD_UNCHANGED)
            height = img.shape[0]
            width = img.shape[1]
            depth = img.shape[2]

            file_loader = FileSystemLoader('template')
            environment = Environment(loader=file_loader, autoescape=True)
            annotation_template = environment.get_template('annotation.xml')

            abspath = os.path.abspath(full_filename)

            template_parameters = {
                'path': abspath,
                'filename': os.path.basename(abspath),
                'folder': os.path.basename(os.path.dirname(abspath)),
                'width': width,
                'height': height,
                'depth': depth,
                'objects': []
            }

            filename_wo_ext = os.path.splitext(filename)[0]

            template_parameters['objects'].append({
                'name': name,
                'result': result,
                'groundtruth': groundtruth,
            })

            with open(os.path.join(file_path, filename_wo_ext) + '.xml', 'w') as file:
                content = annotation_template.render(**template_parameters)
                file.write(content)


if __name__=="__main__":
    result = '&서울 12345'
    TO_XML(file_path, result)
