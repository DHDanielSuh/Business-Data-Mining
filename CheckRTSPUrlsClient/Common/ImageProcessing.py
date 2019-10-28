#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
DB-IDR System main code.
"""
import os
import cv2
import numpy as np
from Common import HoonUtils as hp
import time
import imutils
import operator


def find_four_corners(img):
    normalized_img = normalize_image(img)
    gray = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.medianBlur(gray, 11)
    retval, thresh_gray = cv2.threshold(blurred_image, thresh=240, maxval=255, type=cv2.THRESH_BINARY_INV)
    rsz_img = cv2.resize(thresh_gray, (0, 0), fx=0.3, fy=0.3)
    cv2.imshow("four corners", rsz_img)
    cv2.waitKey(0)
    points = np.argwhere(thresh_gray == 0)
    points = np.fliplr(points)
    x1, y1, w, h = cv2.boundingRect(points)
    x2 = x1 + w
    y2 = y1 + h
    return x1, x2, y1, y2


def line_removal(img):
    """

    :param img:
    :return:
    """

    if img is None:
        print("Image is empty!")
        pass

    hp.hp_imshow(img, desc="Original image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hp.hp_imshow(gray, desc="Gray image")

    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    hp.hp_imshow(bw, desc="Binary image")

    hz = bw
    vt = bw

    hz_size = hz.shape[1] / 30
    hz_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (hz_size, 1, 1))
    hz = cv2.erode(hz, hz_structure, iterations=1)
    hz = cv2.dilate(hz, hz_structure, iterations=1)
    hp.hp_imshow(hz, desc="horizontal")

    vt_size = vt.shape[0] / 30
    vt_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vt_size, 1))
    vt = cv2.erode(vt, vt_structure, iterations=1)
    vt = cv2.dilate(vt, vt_structure, iterations=1)
    hp.hp_imshow(vt, desc="vertical")

    # bitwise_not
    vt = cv2.bitwise_not(vt)
    hp.hp_imshow(vt, desc="vertical bit")

    """
    Extract edges and smooth image according to the logic
    1. extract edges
    2. dilate(edges)
    3. src.copyTo(smooth)
    4. blur smooth img
    5. smooth.copyTo(src, edges)
    """

    edges = cv2.adaptiveThreshold(vt, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
    hp.hp_imshow(edges, desc="edges")

    kernel = np.ones((2, 2), dtype="uint8")
    edges = cv2.dilate(edges, kernel)
    hp.hp_imshow(edges, desc="dilated edges")

    smooth = vt
    smooth = cv2.blur(smooth, (2, 2, 1))
    vt, edges = smooth
    hp.hp_imshow(vt, desc="smooth")


def plot_dots(img, coord_list):
    for coordinates in coord_list:
        cv2.circle(img, coordinates, 15, (15, 15, 255), thickness=10)
    return img


def normalize_image(img):
    min_val = np.min(img)
    max_val = np.max(img)
    normalized_img = ((img - min_val) / (max_val - min_val)) * 255.
    return np.uint8(normalized_img)


def find_black_rects(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
    # rsz_img = cv2.resize(thresh_gray, (0, 0), fx=0.3, fy=0.3)
    # cv2.imshow("result", rsz_img)
    # cv2.waitKey(0)

    cv2.bitwise_not(thresh_gray, thresh_gray)
    _, contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    for cont in contours:
        x1, y1, w, h = cv2.boundingRect(cont)
        k = float(h)/w
        if 0.9 < k < 1.1 and 1000 < w*h < 1500:
            cv2.drawContours(img, contours, i, (0,250,0), thickness=10)

        i += 1
    hp.hp_imshow(img)
    # rsz_img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
    # cv2.imshow("result", rsz_img)
    # cv2.waitKey(0)


def detect_four_corners_based_on_ref_squares(img,
                                             ref_vertex,
                                             search_margin=0.1,
                                             square_width=50./2480,
                                             debug_=False):
    """
    Detect four quadrilateral vertices based on reference black squares.
    It is assumed that four black squares are located
    near the four corners based on reference vertices.

    :param img:
    :param ref_vertex:
    :param search_margin:
    :param square_width:
    :param debug_:
    :return: status, detected vertices, output image
    """
    square_ratio_range = [0.8, 1.2]
    square_width_margin = 0.5
    square_fill_thresh = 0.8

    debug_in_ = False

    dim = img.shape[1::-1]
    square_width = square_width * dim[0]
    real_vertices = hp.generate_four_vertices_from_ref_vertex(ref_vertex, dim)
    crop_boxes = []
    offsets = [int(x*search_margin) for x in dim]
    crop_boxes.append([real_vertices[0][0],            real_vertices[0][1],
                       real_vertices[0][0]+offsets[0], real_vertices[0][1]+offsets[1]])
    crop_boxes.append([real_vertices[1][0]-offsets[0], real_vertices[1][1],
                       real_vertices[1][0],            real_vertices[1][1]+offsets[1]])
    crop_boxes.append([real_vertices[2][0],            real_vertices[2][1]-offsets[1],
                       real_vertices[2][0]+offsets[0], real_vertices[2][1]])
    crop_boxes.append([real_vertices[3][0]-offsets[0], real_vertices[3][1]-offsets[1],
                       real_vertices[3][0],            real_vertices[3][1]])

    detected_vertices = []
    kernel = np.ones((5,5),np.uint8)
    for idx in range(4):
        crop_img = img[crop_boxes[idx][1]:crop_boxes[idx][3],
                       crop_boxes[idx][0]:crop_boxes[idx][2]]
        gray_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
        ret, thresh_gray = cv2.threshold(gray_img,
                                         thresh=200,
                                         maxval=255,
                                         # type=cv2.THRESH_BINARY)
                                         type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        for _ in range(3):
            thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, kernel)
        cv2.bitwise_not(thresh_gray, thresh_gray)
        thresh_color = cv2.cvtColor(thresh_gray, cv2.COLOR_GRAY2RGB)
        ret, contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_width_ratio = 1
        det_vertex = []
        for i, cont in enumerate(contours):
            x1, y1, w, h = cv2.boundingRect(cont)
            ratio = h/w
            width_ratio = abs(square_width - w) / square_width
            if debug_in_:
                if w > 10 and h > 10:
                    # cv2.drawContours(crop_img, contours, i, hp.GREEN, thickness=4)
                    cv2.drawContours(thresh_color, contours, i, hp.GREEN, thickness=4)
                    print("-------")
                    print(i)
                    print(x1, y1, w, h, width_ratio)
                    print(ratio, cv2.contourArea(cont)/(w*h))

            if square_ratio_range[0] < ratio < square_ratio_range[1] and \
               width_ratio < square_width_margin and \
               cv2.contourArea(cont) / (w*h) > square_fill_thresh:
                moments = cv2.moments(cont)
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                if width_ratio < min_width_ratio:
                    det_vertex = [cx + crop_boxes[idx][0], cy + crop_boxes[idx][1]]
                    min_width_ratio = width_ratio
                    # print("****")
                if debug_:
                    disp_img = np.copy(crop_img)
                    cv2.drawContours(disp_img, contours, i, hp.RED, thickness=4)
                    cv2.circle(disp_img,(cx, cy), 8, hp.GREEN, -1)
                    hp.hp_imshow(disp_img)
        if debug_in_:
            hp.hp_imshow(thresh_color, desc="thresh_color")
        if det_vertex:
            detected_vertices.append(det_vertex)
    box_img = np.copy(img)
    if len(detected_vertices) != 4:
        # print(" @ Error: 4 corners are NOT  detected!")
        return False, detected_vertices, img

    cv2.line(box_img, tuple(detected_vertices[0]), tuple(detected_vertices[1]), hp.RED, 10)
    cv2.line(box_img, tuple(detected_vertices[0]), tuple(detected_vertices[2]), hp.RED, 10)
    cv2.line(box_img, tuple(detected_vertices[1]), tuple(detected_vertices[3]), hp.RED, 10)
    cv2.line(box_img, tuple(detected_vertices[2]), tuple(detected_vertices[3]), hp.RED, 10)
    if debug_:
        hp.hp_imshow(box_img, desc="four corners")

    return True, detected_vertices, img


def draw_line_from_rho_and_theta(img, rho, theta, pause_sec=-1):

    img_sz = img.shape[1::-1]
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    x = []
    y = []
    if b != 0:
        slope = -a / b
        y1 = slope * (-x0) + y0
        if 0 <= y1 < img_sz[1]:
            x.append(0)
            y.append(y1)
        y1 = slope * (img_sz[0] - 1 - x0) + y0
        if 0 <= y1 < img_sz[1]:
            x.append(img_sz[0] - 1)
            y.append(y1)
        x1 = (-y0) / slope + x0
        if 0 <= x1 < img_sz[0]:
            x.append(x1)
            y.append(0)
        x1 = (img_sz[1] - 1 - y0) / slope + x0
        if 0 <= x1 < img_sz[0]:
            x.append(x1)
            y.append(img_sz[1] - 1)
    else:
        x = [x0, x0]
        y = [0, img_sz[1]-1]
    angle = (90 - (theta * 180 / np.pi))
    if pause_sec >= 0:
        print(" # rotated angle = {:f} <- ({:.3f}, {:.3f})".format(angle, theta, rho))
    if len(x) is 2:
        pts = [[int(x[0]+0.5), int(y[0]+0.5)], [int(x[1]+0.5), int(y[1]+0.5)]]
    else:
        if pause_sec >= 0:
            print(" @ Warning: rho is zero.\n")
        pts = [[0,0],[0,0]]

    line_img = np.copy(img)
    cv2.line(line_img, (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]), hp.RED, 4)
    hp.hp_imshow(line_img, pause_sec=pause_sec)

    return pts


def check_lines_in_img(img, algorithm='HoughLineTransform'):

    if img.shape != 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_rgb = np.copy(img)
    else:
        img_gray = np.copy(img)
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    ret, img_bw = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_edge = cv2.Canny(img_bw, 50, 150, apertureSize=3)

    if algorithm is 'HoughLineTransform':

        lines = cv2.HoughLines(img_edge, 1, np.pi/180, 100)
        print(" # Total lines: {:d}".format(len(lines)))
        for line in lines:
            img_lines = np.copy(img_rgb)
            dim = img_lines.shape
            rho = line[0][0]
            theta = line[0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x = []
            y = []
            if b != 0:
                slope = -a / b
                y1 = slope * (-x0) + y0
                if 0 <= y1 < dim[0]:
                    x.append(0)
                    y.append(y1)
                y1 = slope * (dim[1] - 1 - x0) + y0
                if 0 <= y1 < dim[0]:
                    x.append(dim[1] - 1)
                    y.append(y1)
                x1 = (-y0) / slope + x0
                if 0 <= x1 < dim[1]:
                    x.append(x1)
                    y.append(0)
                x1 = (dim[0] - 1 - y0) / slope + x0
                if 0 <= x1 < dim[1]:
                    x.append(x1)
                    y.append(dim[0] - 1)
            else:
                x = [x0, x0]
                y = [0, dim[0]-1]
            angle = (90 - (theta * 180 / np.pi))
            print(" # rotated angle = {:.1f} <- ({:f}, {:f})".format(angle, theta, rho))
            if len(x) is 2:
                img_lines = cv2.line(img_rgb, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), hp.RED, 4)
                hp.plt_imshow(img_lines)
                # if -5 < angle < 0 or 0 < angle < 5:
                #     plt_imshow(img_line)
            else:
                print(" @ Warning: something wrong.\n")
                pass

    elif algorithm == 'ProbabilisticHoughTransform':
        end_pts_list = cv2.HoughLinesP(img_edge, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)
        img_lines = np.copy(img_rgb)
        print(" # Total lines: {:d}".format(len(end_pts_list)))
        for end_pts in end_pts_list:
            cv2.line(img_lines, tuple(end_pts[0][0:2]), tuple(end_pts[0][2:]), hp.RED, 10)
            angle = np.arctan2(end_pts[0][3]-end_pts[0][1], end_pts[0][2]-end_pts[0][0]) * 180. / np.pi
            print(" # rotated angle = {:.1f}".format(angle))
            hp.hp_imshow(img_lines)
        # if -5 < angle < 0 or 0 < angle < 5:
        #     plt_imshow(img_line)

    return True


def derotate_image(img,
                   max_angle=30,
                   max_angle_candidates=50,
                   angle_resolution=0.5,
                   inside_margin_ratio=0.1,
                   rot_img_fname=None,
                   check_time_=False,
                   pause_img_sec=-1):
    """
    Derotate image.

    :param img:
    :param max_angle: Maximum rotated angle. The angles above this should be ignored.
    :param max_angle_candidates:
    :param angle_resolution:
    :param inside_margin_ratio:
    :param rot_img_fname:
    :param check_time_:
    :param pause_img_sec:
    :return:
    """
    start_time = None
    if check_time_:
        start_time = time.time()
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img_gray = np.amin(img, axis=2)
    else:
        img_gray = np.copy(img)

    inside_margin = [int(x * inside_margin_ratio) for x in img.shape[1::-1]]

    img_gray[ :inside_margin[1],:] = 255
    img_gray[-inside_margin[1]:,:] = 255
    img_gray[:, :inside_margin[0]] = 255
    img_gray[:,-inside_margin[0]:] = 255

    if False:
        check_lines_in_img(img, algorithm='HoughLineTransform')
        check_lines_in_img(img, algorithm='ProbabilisticHoughTransform')

    ret, img_bw = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    """
    kernel = np.ones((5, 5), np.uint8)  # note this is a horizontal kernel
    bw = np.copy(img_bw)
    for i in range(9):
        bw = cv2.erode(bw, kernel, iterations=1)
        bw = cv2.dilate(bw, kernel, iterations=1)
    hp.hp_imshow(hp.hstack_images((img_bw, bw)))
    """
    img_edge = cv2.Canny(img_bw, 50, 150, apertureSize=3)
    if False:
        hp.hp_imshow(img_edge)
        # hp.plt_imshow(edges)

    lines = cv2.HoughLines(img_edge, 1, np.pi/360, int(min(img_edge.shape)/8.))

    angles = []
    if lines is not None:
        for cnt, line in enumerate(lines):
            angle = int((90 - line[0][1] * 180 / np.pi) / float(angle_resolution)) * angle_resolution
            draw_line_from_rho_and_theta(img, line[0][0], line[0][1], pause_sec=-1)

            if abs(angle) < max_angle:
                angles.append(angle)
            if max_angle_candidates < cnt:
                break

    # rot_angle = max(set(angles), key=angles.count)
    sorted_angles = sorted({x:angles.count(x) for x in angles}.items(), key=operator.itemgetter(1), reverse=True)

    if len(sorted_angles) == 0:
        rot_angle = 0
    elif len(sorted_angles) == 1:
        rot_angle = sorted_angles[0][0]
    elif sorted_angles[0][0] == 0 and (sorted_angles[0][1] < 2 * sorted_angles[1][1]):
        rot_angle = sorted_angles[1][0]
    elif (sorted_angles[0][1] / sorted_angles[1][1]) < 3 and abs(sorted_angles[0][0] - sorted_angles[1][0]) <= 1.0:
        rot_angle = (sorted_angles[0][0] + sorted_angles[1][0]) / 2.
    else:
        rot_angle = sorted_angles[0][0]

    """
    if rot_angle != 0:
        rot_angle += 0.5
    """
    if pause_img_sec >= 0:
        print("# Rotated angle is {:5.1f} degree.".format(rot_angle))

    sz = img_bw.shape[1::-1]
    rot_img = ~imutils.rotate(~img, angle=-rot_angle, center=(int(sz[0]/2), int(sz[1]/2)), scale=1)
    if check_time_:
        print(" # Time for rotation detection and de-rotation if any : {:.2f} sec".
              format(float(time.time() - start_time)))

    if 0 <= pause_img_sec:
        hp.imshow(np.concatenate((img, rot_img), axis=1), pause_sec=pause_img_sec, desc="de-rotation")

    if rot_img_fname:
        hp.hp_imwrite(rot_img_fname, rot_img, 'RGB')

    return rot_img


def erase_lines_in_image(img,
                         check_time_=False,
                         pause_img_sec=-1):
    """
    Erase lines.

    :param img:
    :param check_time_:
    :param pause_img_sec:
    :return:
    """
    erase_window_sz = 9
    line_thresh = 32

    start_time = None
    if check_time_:
        start_time = time.time()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else np.copy(img)
    ret, img_bw = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    """
    kernel = np.ones((1, 5), np.uint8)  # note this is a horizontal kernel
    bw = np.copy(img_bw)
    for i in range(9):
        bw = cv2.dilate(bw, kernel, iterations=1)
        bw = cv2.erode(bw, kernel, iterations=1)
    hp.hp_imshow(hp.hstack_images((img_bw, bw)))
    """

    img_edge = cv2.Canny(img_bw, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(img_edge, 1, np.pi/360, 200)
    print(" # Total number of lines detected is {:d}.".format(len(lines)))
    hp.hp_imshow(img_bw, pause_sec=pause_img_sec)

    img_erase = np.ones((img_bw.shape[:2]), dtype=np.uint8) * 255
    for cnt, line in enumerate(lines):
        line_pts = draw_line_from_rho_and_theta(img, line[0][0], line[0][1], pause_sec=pause_img_sec)
        x0, y0, x1, y1 = line_pts[0][0], line_pts[0][1], line_pts[1][0], line_pts[1][1]
        pnts = []
        if (x1 - x0) > (y1 - y0):
            for x in range(x0, x1):
                y = (y1 - y0) / (x1 - x0) * (x - x0) + y0
                pnts.append([int(x), int(y), img_bw[int(y),int(x)]])
        else:
            for y in range(y0, y1):
                x = (x1 - x0) / (y1 - y0) * (y - y0) + x0
                pnts.append([int(x), int(y), img_bw[int(y),int(x)]])
        cnt = 0
        stt_pnt = 0
        for i in range(len(pnts)):
            if pnts[i][2] == 255:
                if cnt == 0:
                    stt_pnt = i
                cnt += 1
            else:
                if cnt > line_thresh:
                    # print(" > {:d}-th line: {:d} + {:d} = {:d}".format(cnt, stt_pnt, cnt, stt_pnt+cnt))
                    for j in range(cnt):
                        pos = pnts[stt_pnt+j][:2]
                        if (x1 - x0) > (y1 - y0):
                            img_erase[pos[1]-erase_window_sz:pos[1]+erase_window_sz+1,pos[0]] = 0
                        else:
                            img_erase[pos[1], pos[0]-erase_window_sz:pos[0]+erase_window_sz+1] = 0
                cnt = 0
        hp.hp_imshow(img_erase, pause_sec=pause_img_sec)

    if check_time_:
        print(" # The processing time of erasing line function is {:.3f} sec".
              format(float(time.time() - start_time)))

    img_bw_erase = ((img_erase == 0) * 0 + (img_erase != 0) * img_bw).astype(np.uint8)

    return img_erase, img_bw_erase


def run__crop():
    paper_size = 'A4'
    ref_dim = [1280, None]

    org_img = hp.hp_imread("test_videos/DB_1.jpg")
    ret, det_vertices, box_img = detect_four_corners_based_on_ref_squares(org_img,
                                                                          (0,0),
                                                                          search_margin=0.1,
                                                                          square_width=50./2480,
                                                                          debug_=False)
    if paper_size == 'A4':
        ref_dim[1] = int(ref_dim[0] * np.sqrt(2.))

    tar_vertices = [[0,0], [ref_dim[0],0], [0, ref_dim[1]], [ref_dim[0], ref_dim[1]]]
    mtx = cv2.getPerspectiveTransform(np.float32(det_vertices),np.float32(tar_vertices))
    warp_img = cv2.warpPerspective(org_img, mtx, dsize=tuple(ref_dim), flags=cv2.INTER_LINEAR)
    hp.hp_imshow(warp_img, "output")
    hp.hp_imwrite(warp_img, "output.png")


def run__derotate():

    img_path = 'test_videos/census-rotate-2.jpeg'

    imgs, filenames = hp.hp_imread_all_images(img_path,
                                              fname_prefix='census-rotate-')
    for img in imgs:
        rot_img = derotate_image(img,
                                 rot_img_fname=None,
                                 check_time_=False,
                                 pause_img_sec=0)
        hp.hp_imshow(np.hstack((img, rot_img)), desc="de-rotation")


def run__erase_lines_in_image():

    img_path = 'test_videos/census-1.jpg'
    img_prefix = ''
    imgs, filenames = hp.hp_imread_all_images(img_path, fname_prefix=img_prefix)
    for img in imgs:
        erase_img = erase_lines_in_image(img, pause_img_sec=0)
        hp.hp_imshow(np.hstack((cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), erase_img)), desc="erase lines")


def detect_text_area(img, area_ratio_thresh=0.25, char_min_pxl=10, box_img_=False, debug_=False):

    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # smooth the image to avoid noises
    filtered = cv2.medianBlur(gray, 3)

    # Apply adaptive threshold
    # thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
    _, bin = cv2.threshold(filtered, thresh=128, maxval=255, type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5,5), np.uint8)
    """
    kernel[0:2,0:2] = 0
    kernel[0:2,3:5] = 0
    kernel[3:5,3:5] = 0
    kernel[3:5,0:2] = 0
    kernel[1,1] = 1
    kernel[1,3] = 1
    kernel[3,1] = 1
    kernel[3,3] = 1
    """

    morph = np.copy(bin)
    for _ in range(5):
        morph = cv2.dilate(morph, kernel, iterations=3)
        morph = cv2.erode( morph, kernel, iterations=3)

    # Find the contours
    image, contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, find the bounding rectangle and draw it
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        rect_area = w * h
        if (contour_area / rect_area > area_ratio_thresh) and w > char_min_pxl and h > char_min_pxl:
            boxes.append([x,y,x+w,y+h])

    boxes_img = []
    if debug_:
        boxes_img = hp.draw_boxes_on_img(np.copy(img), boxes, color=hp.RED, thickness=4)
        hp.imshow(morph)
        hp.imshow(boxes_img)

    return boxes, boxes_img


def check_bw_image(image, black_thresh=16, white_thresh=240):
    black = sum(sum(image < black_thresh))
    white = sum(sum(image > white_thresh))

    pxl_sum = 0
    if len(black) == 3:
        if not black[0] == black[1] == black[2]:
            return False
        pxl_sum += black[0]
    if len(white) == 3:
        if not white[0] == white[1] == white[2]:
            return False
        pxl_sum += white[0]

    sz = image.shape
    num_pxl = sz[0] * sz[1]
    if num_pxl == pxl_sum:
        return True
    else:
        return False


def crop_images(path_in, path_out, num_images):
    filename = os.path.splitext(os.path.basename(path_in))[0]

    img = cv2.imread(path_in,0)
    img2 = cv2.cvtColor(img, cv.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(img2)
    print(img.shape)
    print(img)
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:num_images]
    mask = np.zeros_like(img)
    for cnt in cnts:
        print(cnt.shape)
        mask = np.zeros_like(img)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        cv2.drawContours(mask, [approx], -1, (255, 0, 0), -1)

        x,y,w,h = cv2.boundingRect(cnt)
        print(x, y, w, h)

        mask[y:y+h, x:x+w] = 255
        cv2.rectangle(mask,(x,y),(x+w,y+h),(255,255,255),2)

        out = np.zeros_like(img)
        out[mask == 255] = img[mask == 255]

        (x, y, _) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        out = out[topx:bottomx + 1, topy:bottomy + 1]

        img_name = filename + str(i) + '.jpeg'
        cv2.imwrite(os.path.join(path_out, img_name), out)

        print(filename + ': ' + str(i))

    cv2.imwrite(path_out, mask)


def reduce_img_resolution(img, scale_percent):

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized


if __name__ == "__main__":
    # run__crop()
    run__derotate()
    # run__erase_lines_in_image()
