import os
import shutil


def create_filename_list(base_dir, save_path, start_idx, end_idx, ext='.jpg'):
    f = open(save_path, 'w+')

    for i in range(start_idx, end_idx + 1):
        f.write(os.path.join(base_dir, str(i) + ext) + '\n')

    f.close()


def combine_xml_jpg_files(input_dir,
                          output_dir_img, output_dir_xml,
                          img_dir='Images',
                          img_ext='.jpg', xml_ext='.xml',
                          input_file_ext='.xml'):
    if not os.path.exists(output_dir_img):
        os.makedirs(output_dir_img)

    if not os.path.exists(output_dir_xml):
        os.makedirs(output_dir_xml)

    i = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(input_file_ext):
                src_path_xml = os.path.join(root, file)
                dst_path_xml = os.path.join(output_dir_xml, str(i) + xml_ext)

                src_path_img = os.path.join(*root.split('/')[:-1], img_dir, file.split('.')[0] + img_ext)
                if root[0] == '/':
                    src_path_img = os.path.join('/', src_path_img)
                dst_path_img = os.path.join(output_dir_img, str(i) + img_ext)

                shutil.copyfile(src_path_xml, dst_path_xml)
                shutil.copyfile(src_path_img, dst_path_img)

                i += 1


def save_result_file(obj_arr, img_width, img_height, save_path):
    f = open(save_path, 'w')

    for obj in obj_arr:

        obj_code = obj['object']

        confidence = obj['confidence']

        x = obj['pos'][0] / img_width
        y = obj['pos'][1] / img_height
        w = (obj['pos'][2] - x) / img_width
        h = (obj['pos'][3] - y) / img_height

        f.write("{0:d} {1:.3f} {2:.3f} {3:.3f} {4:.3f} {5:.3f}\n".format(obj_code, confidence, x, y, w, h))

