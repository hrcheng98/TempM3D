import io as sysio
import time
import math
import numba
import numpy as np
from numba import cuda


import os
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# AP_mode = 40
AP_mode = 11


@cuda.jit(
    device=True,
    inline=True)
def line_segment_intersection_v1(pts1, pts2, i, j, temp_pts):
    a = cuda.local.array((2, ), dtype=numba.float32)
    b = cuda.local.array((2, ), dtype=numba.float32)
    c = cuda.local.array((2, ), dtype=numba.float32)
    d = cuda.local.array((2, ), dtype=numba.float32)

    a[0] = pts1[2 * i]
    a[1] = pts1[2 * i + 1]

    b[0] = pts1[2 * ((i + 1) % 4)]
    b[1] = pts1[2 * ((i + 1) % 4) + 1]

    c[0] = pts2[2 * j]
    c[1] = pts2[2 * j + 1]

    d[0] = pts2[2 * ((j + 1) % 4)]
    d[1] = pts2[2 * ((j + 1) % 4) + 1]

    area_abc = trangle_area(a, b, c)
    area_abd = trangle_area(a, b, d)

    if area_abc * area_abd >= 0:
        return False

    area_cda = trangle_area(c, d, a)
    area_cdb = area_cda + area_abc - area_abd

    if area_cda * area_cdb >= 0:
        return False
    t = area_cda / (area_abd - area_abc)

    dx = t * (b[0] - a[0])
    dy = t * (b[1] - a[1])
    temp_pts[0] = a[0] + dx
    temp_pts[1] = a[1] + dy
    return True

@cuda.jit(
    device=True,
    inline=True)
def line_segment_intersection(pts1, pts2, i, j, temp_pts):
    A = cuda.local.array((2, ), dtype=numba.float32)
    B = cuda.local.array((2, ), dtype=numba.float32)
    C = cuda.local.array((2, ), dtype=numba.float32)
    D = cuda.local.array((2, ), dtype=numba.float32)

    A[0] = pts1[2 * i]
    A[1] = pts1[2 * i + 1]

    B[0] = pts1[2 * ((i + 1) % 4)]
    B[1] = pts1[2 * ((i + 1) % 4) + 1]

    C[0] = pts2[2 * j]
    C[1] = pts2[2 * j + 1]

    D[0] = pts2[2 * ((j + 1) % 4)]
    D[1] = pts2[2 * ((j + 1) % 4) + 1]
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            temp_pts[0] = Dx / DH
            temp_pts[1] = Dy / DH
            return True
    return False

@cuda.jit(device=True, inline=True)
def point_in_quadrilateral(pt_x, pt_y, corners):
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]

    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]

    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]

    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1

    eps = -1e-6
    return abab - abap >= eps and abap >= eps and adad - adap >= eps and adap >= eps

@cuda.jit(device=True, inline=True)
def quadrilateral_intersection(pts1, pts2, int_pts):
    num_of_inter = 0
    for i in range(4):
        if point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    temp_pts = cuda.local.array((2, ), dtype=numba.float32)
    for i in range(4):
        for j in range(4):
            has_pts = line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1

    return num_of_inter

@cuda.jit(device=True, inline=True)
def rbbox_to_corners(corners, rbbox):
    # generate clockwise corners and rotate it clockwise
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = cuda.local.array((4, ), dtype=numba.float32)
    corners_y = cuda.local.array((4, ), dtype=numba.float32)
    corners_x[0] = -x_d / 2
    corners_x[1] = -x_d / 2
    corners_x[2] = x_d / 2
    corners_x[3] = x_d / 2
    corners_y[0] = -y_d / 2
    corners_y[1] = y_d / 2
    corners_y[2] = y_d / 2
    corners_y[3] = -y_d / 2
    for i in range(4):
        corners[2 * i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i +
                1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y


@cuda.jit(device=True, inline=True)
def sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    if num_of_inter > 0:
        center = cuda.local.array((2, ), dtype=numba.float32)
        center[:] = 0.0
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        v = cuda.local.array((2, ), dtype=numba.float32)
        vs = cuda.local.array((16, ), dtype=numba.float32)
        for i in range(num_of_inter):
            v[0] = int_pts[2 * i] - center[0]
            v[1] = int_pts[2 * i + 1] - center[1]
            d = math.sqrt(v[0] * v[0] + v[1] * v[1])
            v[0] = v[0] / d
            v[1] = v[1] / d
            if v[1] < 0:
                v[0] = -2 - v[0]
            vs[i] = v[0]
        j = 0
        temp = 0
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2 * i]
                ty = int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1

                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty

@cuda.jit(device=True, inline=True)
def trangle_area(a, b, c):
    return (
        (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0

@cuda.jit(device=True, inline=True)
def area(int_pts, num_of_inter):
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(
            trangle_area(int_pts[:2], int_pts[2 * i + 2:2 * i + 4],
                         int_pts[2 * i + 4:2 * i + 6]))
    return area_val

@cuda.jit(device=True, inline=True)
def inter(rbbox1, rbbox2):
    corners1 = cuda.local.array((8, ), dtype=numba.float32)
    corners2 = cuda.local.array((8, ), dtype=numba.float32)
    intersection_corners = cuda.local.array((16, ), dtype=numba.float32)

    rbbox_to_corners(corners1, rbbox1)
    rbbox_to_corners(corners2, rbbox2)

    num_intersection = quadrilateral_intersection(corners1, corners2,
                                                  intersection_corners)
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)
    # print(intersection_corners.reshape([-1, 2])[:num_intersection])

    return area(intersection_corners, num_intersection)


@cuda.jit('(float32[:], float32[:], int32)', device=True, inline=True)
def devRotateIoUEval(rbox1, rbox2, criterion=-1):
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = inter(rbox1, rbox2)
    if criterion == -1:
        return area_inter / (area1 + area2 - area_inter)
    elif criterion == 0:
        return area_inter / area1
    elif criterion == 1:
        return area_inter / area2
    else:
        return area_inter


@cuda.jit(
    '(int64, int64, float32[:], float32[:], float32[:], int32)',
    fastmath=False)
def rotate_iou_kernel_eval(N,
                           K,
                           dev_boxes,
                           dev_query_boxes,
                           dev_iou,
                           criterion=-1):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.x
    col_start = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    row_size = min(N - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(K - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(64 * 5, ), dtype=numba.float32)
    block_qboxes = cuda.shared.array(shape=(64 * 5, ), dtype=numba.float32)

    dev_query_box_idx = threadsPerBlock * col_start + tx
    dev_box_idx = threadsPerBlock * row_start + tx
    if (tx < col_size):
        block_qboxes[tx * 5 + 0] = dev_query_boxes[dev_query_box_idx * 5 + 0]
        block_qboxes[tx * 5 + 1] = dev_query_boxes[dev_query_box_idx * 5 + 1]
        block_qboxes[tx * 5 + 2] = dev_query_boxes[dev_query_box_idx * 5 + 2]
        block_qboxes[tx * 5 + 3] = dev_query_boxes[dev_query_box_idx * 5 + 3]
        block_qboxes[tx * 5 + 4] = dev_query_boxes[dev_query_box_idx * 5 + 4]
    if (tx < row_size):
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0]
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1]
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2]
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3]
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4]
    cuda.syncthreads()
    if tx < row_size:
        for i in range(col_size):
            offset = row_start * threadsPerBlock * K + col_start * threadsPerBlock + tx * K + i
            dev_iou[offset] = devRotateIoUEval(block_qboxes[i * 5:i * 5 + 5],
                                               block_boxes[tx * 5:tx * 5 + 5],
                                               criterion)

@numba.jit(nopython=True)
def div_up(m, n):
    return m // n + (m % n > 0)


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """rotated box iou running in gpu. 8x faster than cpu version
    (take 5ms in one example with numba.cuda code).
    convert from [this project](
        https://github.com/hongzhenwang/RRPN-revise/tree/master/lib/rotation).

    Args:
        boxes (float tensor: [N, 5]): rbboxes. format: centers, dims,
            angles(clockwise when positive)
        query_boxes (float tensor: [K, 5]): [description]
        device_id (int, optional): Defaults to 0. [description]

    Returns:
        [type]: [description]
    """
    box_dtype = boxes.dtype
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return iou
    threadsPerBlock = 8 * 8
    cuda.select_device(device_id)
    blockspergrid = (div_up(N, threadsPerBlock), div_up(K, threadsPerBlock))

    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes.reshape([-1]), stream)
        query_boxes_dev = cuda.to_device(query_boxes.reshape([-1]), stream)
        iou_dev = cuda.to_device(iou.reshape([-1]), stream)
        rotate_iou_kernel_eval[blockspergrid, threadsPerBlock, stream](
            N, K, boxes_dev, query_boxes_dev, iou_dev, criterion)
        iou_dev.copy_to_host(iou.reshape([-1]), stream=stream)
    return iou.astype(boxes.dtype)


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    # print(len(thresholds), len(scores), num_gt)
    return thresholds


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    CLASS_NAMES = [
        'car', 'pedestrian', 'cyclist', 'van', 'person_sitting', 'car',
        'tractor', 'trailer'
    ]
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        elif (current_cls_name == "Pedestrian".lower()
              and "Person_sitting".lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == "Car".lower() and "Van".lower() == gt_name):
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if ((gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
                or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])
                or (height <= MIN_HEIGHT[difficulty])):
            # if gt_anno["difficulty"][i] > difficulty or gt_anno["difficulty"][i] == -1:
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
        # for i in range(num_gt):
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    for i in range(num_dt):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(
                    boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                                (boxes[n, 2] - boxes[n, 0]) *
                                (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1, stable=True):
    # riou = box_np_ops.riou_cc(boxes, qboxes)
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
def box3d_overlap_kernel(boxes,
                         qboxes,
                         rinc,
                         criterion=-1,
                         z_axis=1,
                         z_center=1.0):
    """
        z_axis: the z (height) axis.
        z_center: unified z (height) center of box.
    """
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                min_z = min(
                    boxes[i, z_axis] + boxes[i, z_axis + 3] * (1 - z_center),
                    qboxes[j, z_axis] + qboxes[j, z_axis + 3] * (1 - z_center))
                max_z = max(
                    boxes[i, z_axis] - boxes[i, z_axis + 3] * z_center,
                    qboxes[j, z_axis] - qboxes[j, z_axis + 3] * z_center)
                iw = min_z - max_z
                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = 1.0
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def box3d_overlap(boxes, qboxes, criterion=-1, z_axis=1, z_center=1.0):
    """kitti camera format z_axis=1.
    """
    bev_axes = list(range(7))
    bev_axes.pop(z_axis + 3)
    bev_axes.pop(z_axis)

    # t = time.time()
    # rinc = box_np_ops.rinter_cc(boxes[:, bev_axes], qboxes[:, bev_axes])
    rinc = rotate_iou_gpu_eval(boxes[:, bev_axes], qboxes[:, bev_axes], 2)
    # print("riou time", time.time() - t)
    box3d_overlap_kernel(boxes, qboxes, rinc, criterion, z_axis, z_center)
    return rinc


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]
    # gt_bboxes = gt_datas[:, :4]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size,))
    thresh_idx = 0
    delta = np.zeros((gt_size,))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            # only a tp add a threshold.
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if (assigned_detection[j]):
                        continue
                    if (ignored_det[j] == -1 or ignored_det[j] == 1):
                        continue
                    if (ignored_threshold[j]):
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx,))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:gt_num +
                                                                  gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(gt_annos,
                         dt_annos,
                         metric,
                         num_parts=50,
                         z_axis=1,
                         z_center=1.0):
    """fast iou algorithm. this function can be used independently to
    do result analysis.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
        z_axis: height axis. kitti camera use 1, lidar use 2.
    """
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0
    bev_axes = list(range(3))
    bev_axes.pop(z_axis)
    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate(
                [a["location"][:, bev_axes] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, bev_axes] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            loc = np.concatenate(
                [a["location"][:, bev_axes] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, bev_axes] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            overlap_part = bev_box_overlap(gt_boxes,
                                           dt_boxes).astype(np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1)
            overlap_part = box3d_overlap(
                gt_boxes, dt_boxes, z_axis=z_axis,
                z_center=z_center).astype(np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx +
                                              gt_box_num, dt_num_idx:dt_num_idx +
                                                                     dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate(
            [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1)
        dt_datas = np.concatenate([
            dt_annos[i]["bbox"], dt_annos[i]["alpha"][..., np.newaxis],
            dt_annos[i]["score"][..., np.newaxis]
        ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)


def eval_class_v3(gt_annos,
                  dt_annos,
                  current_classes,
                  difficultys,
                  metric,
                  min_overlaps,
                  compute_aos=False,
                  z_axis=1,
                  z_center=1.0,
                  num_parts=50):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_class: int, 0: car, 1: pedestrian, 2: cyclist
        difficulty: int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlap: float, min overlap. official:
            [[0.7, 0.5, 0.5], [0.7, 0.5, 0.5], [0.7, 0.5, 0.5]]
            format: [metric, class]. choose one from matrix above.
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(
        dt_annos,
        gt_annos,
        metric,
        num_parts,
        z_axis=z_axis,
        z_center=z_center)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    precision = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    all_thresholds = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    for m, current_class in enumerate(current_classes):
        for l, difficulty in enumerate(difficultys):
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []
                for i in range(len(gt_annos)):
                    rets = compute_statistics_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        metric,
                        min_overlap=min_overlap,
                        thresh=0.0,
                        compute_fp=False)
                    tp, fp, fn, similarity, thresholds = rets
                    thresholdss += thresholds.tolist()
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                # print(thresholds)
                all_thresholds[m, l, k, :len(thresholds)] = thresholds
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx:idx + num_part], 0)
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx:idx + num_part], 0)
                    dc_datas_part = np.concatenate(
                        dontcares[idx:idx + num_part], 0)
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx:idx + num_part], 0)
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx:idx + num_part], 0)
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos)
                    idx += num_part
                for i in range(len(thresholds)):
                    # recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(
                        precision[m, l, k, i:], axis=-1)
                    # recall[m, l, k, i] = np.max(recall[m, l, k, :i + 1], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
                # use interp to calculate recall
                """
                current_recalls = np.linspace(0, 1, 41)
                prec_unique, inds = np.unique(precision[m, l, k], return_index=True)
                current_recalls = current_recalls[inds]
                f = interp1d(prec_unique, current_recalls)
                precs_for_recall = np.linspace(0, 1, 41)
                max_prec = np.max(precision[m, l, k])
                valid_prec = precs_for_recall < max_prec
                num_valid_prec = valid_prec.sum()
                recall[m, l, k, :num_valid_prec] = f(precs_for_recall[valid_prec])
                """
    ret_dict = {
        "recall": recall,  # [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]
        "precision": precision,
        "orientation": aos,
        "thresholds": all_thresholds,
        "min_overlaps": min_overlaps,
    }
    return ret_dict


def get_mAP(prec):
    sums = 0
    if AP_mode == 40:
        # for i in range(0, prec.shape[-1], 1):
        for i in range(1, prec.shape[-1], 1):
            sums = sums + prec[..., i]
        return sums / 40 * 100


    if AP_mode == 11:
        for i in range(0, prec.shape[-1], 4):
            sums = sums + prec[..., i]
        return sums / 11 * 100


def do_eval_v2(gt_annos,
               dt_annos,
               current_classes,
               min_overlaps,
               compute_aos=False,
               difficultys=(0, 1, 2),
               z_axis=1,
               z_center=1.0):
    # min_overlaps: [num_minoverlap, metric, num_class]
    ret = eval_class_v3(
        gt_annos,
        dt_annos,
        current_classes,
        difficultys,
        0,
        min_overlaps,
        compute_aos,
        z_axis=z_axis,
        z_center=z_center)
    # ret: [num_class, num_diff, num_minoverlap, num_sample_points]
    mAP_bbox = get_mAP(ret["precision"])
    mAP_aos = None
    if compute_aos:
        mAP_aos = get_mAP(ret["orientation"])
    ret = eval_class_v3(
        gt_annos,
        dt_annos,
        current_classes,
        difficultys,
        1,
        min_overlaps,
        z_axis=z_axis,
        z_center=z_center)
    mAP_bev = get_mAP(ret["precision"])
    ret = eval_class_v3(
        gt_annos,
        dt_annos,
        current_classes,
        difficultys,
        2,
        min_overlaps,
        z_axis=z_axis,
        z_center=z_center)
    mAP_3d = get_mAP(ret["precision"])
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos


def do_eval_v3(gt_annos,
               dt_annos,
               current_classes,
               min_overlaps,
               compute_aos=False,
               difficultys=(0, 1, 2),
               z_axis=1,
               z_center=1.0):
    # min_overlaps: [num_minoverlap, metric, num_class]
    types = ["bbox", "bev", "3d"]
    metrics = {}
    for i in range(3):
        ret = eval_class_v3(
            gt_annos,
            dt_annos,
            current_classes,
            difficultys,
            i,
            min_overlaps,
            compute_aos,
            z_axis=z_axis,
            z_center=z_center)
        metrics[types[i]] = ret
    return metrics


def do_coco_style_eval(gt_annos,
                       dt_annos,
                       current_classes,
                       overlap_ranges,
                       compute_aos,
                       z_axis=1,
                       z_center=1.0):
    # overlap_ranges: [range, metric, num_class]
    min_overlaps = np.zeros([10, *overlap_ranges.shape[1:]])
    for i in range(overlap_ranges.shape[1]):
        for j in range(overlap_ranges.shape[2]):
            min_overlaps[:, i, j] = np.linspace(*overlap_ranges[:, i, j])
    mAP_bbox, mAP_bev, mAP_3d, mAP_aos = do_eval_v2(
        gt_annos,
        dt_annos,
        current_classes,
        min_overlaps,
        compute_aos,
        z_axis=z_axis,
        z_center=z_center)
    # ret: [num_class, num_diff, num_minoverlap]
    mAP_bbox = mAP_bbox.mean(-1)
    mAP_bev = mAP_bev.mean(-1)
    mAP_3d = mAP_3d.mean(-1)
    if mAP_aos is not None:
        mAP_aos = mAP_aos.mean(-1)
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def get_official_eval_result(gt_annos,
                             dt_annos,
                             current_classes,
                             difficultys=[0, 1, 2],
                             z_axis=1,
                             z_center=1.0):
    """
        gt_annos and dt_annos must contains following keys:
        [bbox, location, dimensions, rotation_y, score]
    """
    overlap_mod = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7]])
    overlap_easy = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5],
                             [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5],
                             [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5]])
    # min_overlaps = np.stack([overlap_mod, overlap_easy], axis=0)  # [2, 3, 5]
    overlap_easy2 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5],
                             [0.3, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5],
                             [0.3, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5]])
    min_overlaps = np.stack([overlap_mod, overlap_easy, overlap_easy2], axis=0)
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
        5: 'car',
        6: 'tractor',
        7: 'trailer',
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    metrics = do_eval_v3(
        gt_annos,
        dt_annos,
        current_classes,
        min_overlaps,
        compute_aos,
        difficultys,
        z_axis=z_axis,
        z_center=z_center)
    detail = {}
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        class_name = class_to_name[curcls]
        detail[class_name] = {}
        for i in range(min_overlaps.shape[0]):
            mAPbbox = get_mAP(metrics["bbox"]["precision"][j, :, i])
            mAPbev = get_mAP(metrics["bev"]["precision"][j, :, i])
            mAP3d = get_mAP(metrics["3d"]["precision"][j, :, i])
            detail[class_name][f"bbox@{min_overlaps[i, 0, j]:.2f}"] = mAPbbox.tolist()
            detail[class_name][f"bev@{min_overlaps[i, 1, j]:.2f}"] = mAPbev.tolist()
            detail[class_name][f"3d@{min_overlaps[i, 2, j]:.2f}"] = mAP3d.tolist()

            result += print_str(
                (f"{class_to_name[curcls]} " 
                 "AP(Average Precision)@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            mAPbbox = ", ".join(f"{v:.2f}" for v in mAPbbox)
            mAPbev = ", ".join(f"{v:.2f}" for v in mAPbev)
            mAP3d = ", ".join(f"{v:.2f}" for v in mAP3d)
            result += print_str(f"bbox AP:{mAPbbox}")
            result += print_str(f"bev  AP:{mAPbev}")
            result += print_str(f"3d   AP:{mAP3d}")
            if compute_aos:
                mAPaos = get_mAP(metrics["bbox"]["orientation"][j, :, i])
                detail[class_name][f"aos"] = mAPaos.tolist()
                mAPaos = ", ".join(f"{v:.2f}" for v in mAPaos)
                result += print_str(f"aos  AP:{mAPaos}")
    return {
        "result": result,
        "detail": detail,
    }


def get_coco_eval_result(gt_annos,
                         dt_annos,
                         current_classes,
                         z_axis=1,
                         z_center=1.0):
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
        5: 'car',
        6: 'tractor',
        7: 'trailer',
    }
    class_to_range = {
        0: [0.5, 1.0, 0.05],
        1: [0.25, 0.75, 0.05],
        2: [0.25, 0.75, 0.05],
        3: [0.5, 1.0, 0.05],
        4: [0.25, 0.75, 0.05],
        5: [0.5, 1.0, 0.05],
        6: [0.5, 1.0, 0.05],
        7: [0.5, 1.0, 0.05],
    }
    class_to_range = {
        0: [0.5, 0.95, 10],
        1: [0.25, 0.7, 10],
        2: [0.25, 0.7, 10],
        3: [0.5, 0.95, 10],
        4: [0.25, 0.7, 10],
        5: [0.5, 0.95, 10],
        6: [0.5, 0.95, 10],
        7: [0.5, 0.95, 10],
    }

    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    overlap_ranges = np.zeros([3, 3, len(current_classes)])
    for i, curcls in enumerate(current_classes):
        overlap_ranges[:, :, i] = np.array(
            class_to_range[curcls])[:, np.newaxis]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    mAPbbox, mAPbev, mAP3d, mAPaos = do_coco_style_eval(
        gt_annos,
        dt_annos,
        current_classes,
        overlap_ranges,
        compute_aos,
        z_axis=z_axis,
        z_center=z_center)
    detail = {}
    for j, curcls in enumerate(current_classes):
        class_name = class_to_name[curcls]
        detail[class_name] = {}
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        o_range = np.array(class_to_range[curcls])[[0, 2, 1]]
        o_range[1] = (o_range[2] - o_range[0]) / (o_range[1] - 1)
        result += print_str((f"{class_to_name[curcls]} "
                             "coco AP@{:.2f}:{:.2f}:{:.2f}:".format(*o_range)))
        result += print_str((f"bbox AP:{mAPbbox[j, 0]:.2f}, "
                             f"{mAPbbox[j, 1]:.2f}, "
                             f"{mAPbbox[j, 2]:.2f}"))
        result += print_str((f"bev  AP:{mAPbev[j, 0]:.2f}, "
                             f"{mAPbev[j, 1]:.2f}, "
                             f"{mAPbev[j, 2]:.2f}"))
        result += print_str((f"3d   AP:{mAP3d[j, 0]:.2f}, "
                             f"{mAP3d[j, 1]:.2f}, "
                             f"{mAP3d[j, 2]:.2f}"))
        detail[class_name][f"bbox"] = mAPbbox[j].tolist()
        detail[class_name][f"bev"] = mAPbev[j].tolist()
        detail[class_name][f"3d"] = mAP3d[j].tolist()

        if compute_aos:
            detail[class_name][f"aos"] = mAPaos[j].tolist()
            result += print_str((f"aos  AP:{mAPaos[j, 0]:.2f}, "
                                 f"{mAPaos[j, 1]:.2f}, "
                                 f"{mAPaos[j, 2]:.2f}"))
    return {
        "result": result,
        "detail": detail,
    }


"""
detection
When you want to eval your own dataset, you MUST set correct
the z axis and box z center.
If you want to eval by my KITTI eval function, you must 
provide the correct format annotations.
ground_truth_annotations format:
{
    bbox: [N, 4], if you fill fake data, MUST HAVE >25 HEIGHT!!!!!!
    alpha: [N], you can use -10 to ignore it.
    occluded: [N], you can use zero.
    truncated: [N], you can use zero.
    name: [N]
    location: [N, 3] center of 3d box.
    dimensions: [N, 3] dim of 3d box.
    rotation_y: [N] angle.
}
all fields must be filled, but some fields can fill
zero.
"""


def calc_iou(box1, box2):
    # area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    # area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    xx1 = np.maximum(box1[0], box2[0])
    yy1 = np.maximum(box1[1], box2[1])
    xx2 = np.minimum(box1[2], box2[2])
    yy2 = np.minimum(box1[3], box2[3])

    # w = np.maximum(0.0, xx2 - xx1 + 1)
    # h = np.maximum(0.0, yy2 - yy1 + 1)
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    ovr = inter / (area1 + area2 - inter)

    return ovr

def project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
    """
    Projects a 3D box into 2D vertices

    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
    """

    # compute rotational matrix around yaw axis
    R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                  [0, 1, 0],
                  [-math.sin(ry3d), 0, +math.cos(ry3d)]])

    # 3D bounding box corners
    x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
    y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
    z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])

    x_corners += -l3d / 2
    y_corners += -h3d / 2
    z_corners += -w3d / 2

    # bounding box in object co-ordinate
    corners_3d = np.array([x_corners, y_corners, z_corners])

    # rotate
    corners_3d = R.dot(corners_3d)

    # translate
    corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
    corners_2D = p2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]

    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

    verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T

    if return_3d:
        return verts3d, corners_3d
    else:
        return verts3d

def refine_score_2d_3d(det_path, calib_path='/private/pengliang/KITTI3D/training/calib'):
    # from . import adjust_hy

    det_f = np.loadtxt(det_path, dtype=str).reshape(-1, 16)
    det_f = det_f[det_f[:, 0] == 'Car']

    # load calib files
    with open(os.path.join(calib_path, os.path.basename(det_path)), encoding='utf-8') as f:
        text = f.readlines()
        P2 = np.array(text[2].split(' ')[1:], dtype=np.float32).reshape(3, 4)


    for i in range(len(det_f)):
        bbox2d = det_f[i, 4:8].astype(np.float32)

        val1 = det_f[i, 8:].astype(np.float32)
        xyz_real = det_f[i, -5:-2].astype(np.float32)
        xyz_real[1] -= det_f[i, 8].astype(np.float32) / 2

        p_2d = project_3d(P2, xyz_real[0], xyz_real[1], xyz_real[2], val1[1], val1[0], val1[2], val1[6])
        v_2d = np.array([np.min(p_2d[:, 0]), np.min(p_2d[:, 1]), np.max(p_2d[:, 0]), np.max(p_2d[:, 1])])
        iou_2d = calc_iou(bbox2d, v_2d)

        det_f[i, -1] = iou_2d / np.exp(np.sqrt(det_f[i, 13].astype(np.float32) ** 2 +
                                                   det_f[i, 11].astype(np.float32) ** 2
                                                   ) / 80) \
                          * det_f[i, -1].astype(np.float32)
    return det_f


# def calc_iou(box1, box2):
#     # area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
#     # area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
#     area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
#
#     xx1 = np.maximum(box1[0], box2[0])
#     yy1 = np.maximum(box1[1], box2[1])
#     xx2 = np.minimum(box1[2], box2[2])
#     yy2 = np.minimum(box1[3], box2[3])
#
#     # w = np.maximum(0.0, xx2 - xx1 + 1)
#     # h = np.maximum(0.0, yy2 - yy1 + 1)
#     w = np.maximum(0.0, xx2 - xx1)
#     h = np.maximum(0.0, yy2 - yy1)
#     inter = w * h
#     ovr = inter / (area1 + area2 - inter)
#
#     return ovr
#
# def project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
#     """
#     Projects a 3D box into 2D vertices
#
#     Args:
#         p2 (nparray): projection matrix of size 4x3
#         x3d: x-coordinate of center of object
#         y3d: y-coordinate of center of object
#         z3d: z-cordinate of center of object
#         w3d: width of object
#         h3d: height of object
#         l3d: length of object
#         ry3d: rotation w.r.t y-axis
#     """
#
#     # compute rotational matrix around yaw axis
#     R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
#                   [0, 1, 0],
#                   [-math.sin(ry3d), 0, +math.cos(ry3d)]])
#
#     # 3D bounding box corners
#     x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
#     y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
#     z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])
#
#     x_corners += -l3d / 2
#     y_corners += -h3d / 2
#     z_corners += -w3d / 2
#
#     # bounding box in object co-ordinate
#     corners_3d = np.array([x_corners, y_corners, z_corners])
#
#     # rotate
#     corners_3d = R.dot(corners_3d)
#
#     # translate
#     corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))
#
#     corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
#     corners_2D = p2.dot(corners_3D_1)
#     corners_2D = corners_2D / corners_2D[2]
#
#     bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
#
#     verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T
#
#     if return_3d:
#         return verts3d, corners_3d
#     else:
#         return verts3d
#
#
#
def post_3d(det_dir, target_dir, calib_dir = '/private/personal/pengliang/KITTI3D/testing/calib'):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for i in tqdm(sorted(os.listdir(det_dir))):
        pred = np.loadtxt(os.path.join(det_dir, i), dtype=str).reshape(-1, 16)
        with open(os.path.join(calib_dir, i), encoding='utf-8') as f:
            text = f.readlines()
            P2 = np.array(text[2].split(' ')[1:], dtype=np.float32).reshape(3, 4)

        if pred.shape[0] > 0:
            for j in range(len(pred)):
                bbox2d = pred[j, 4:8].astype(np.float32)

                # '''mod conf'''
                # val1 = pred[j, 8:].astype(np.float32)
                # p_2d = project_3d(P2, val1[3], val1[4] - val1[0] / 2, val1[5],
                #                   val1[1], val1[0], val1[2], val1[6])
                #
                # m_2d = np.array([bbox2d[0], np.min(p_2d[:, 1]), bbox2d[2], np.max(p_2d[:, 1])])
                # iou_2d = calc_iou(bbox2d, m_2d)
                # pred[j, -1] = iou_2d / np.exp(np.sqrt(pred[j, 13].astype(np.float32) ** 2 +
                #                                                pred[j, 11].astype(np.float32) ** 2)/80) * \
                #               pred[j, -1].astype(np.float32)

                '''modify y'''
                val1 = pred[j, 8:].astype(np.float32)
                # range_y = np.arange(-0.1, 0.101, 0.02)
                # range_y = np.arange(-0.2, 0.201, 0.02)
                # range_y = np.arange(-0.3, 0.301, 0.02)
                range_y = np.arange(-0.5, 0.501, 0.02)
                # range_y = np.arange(-0.2, 0.201, 0.01)
                tmp_box = []
                for k in range_y:
                    p_2d = project_3d(P2, val1[3], val1[4]-val1[0]/2+k, val1[5],
                                                    val1[1], val1[0], val1[2], val1[6])

                    m_2d = np.array([bbox2d[0], np.min(p_2d[:, 1]), bbox2d[2], np.max(p_2d[:, 1])])
                    iou_2d = np.abs((np.min(p_2d[:, 1]) - bbox2d[1]) - (bbox2d[3] - np.max(p_2d[:, 1])))
                    tmp_box.append(iou_2d)

                good_k = range_y[np.argmin(np.array(tmp_box))]
                pred[j, 12] = pred[j, 12].astype(np.float32) + good_k

                '''modify h'''
                val1 = pred[j, 8:].astype(np.float32)
                # range_h = np.arange(-0.2, 0.2001, 0.01)
                range_h = np.arange(-0.2, 0.2001, 0.02)
                tmp_box = []
                for k in range_h:
                    p_2d = project_3d(P2, val1[3], val1[4] - val1[0] / 2, val1[5],
                                                    val1[1], val1[0] + k, val1[2], val1[6])

                    m_2d = np.array([bbox2d[0], np.min(p_2d[:, 1]), bbox2d[2], np.max(p_2d[:, 1])])
                    iou_2d = calc_iou(bbox2d, m_2d)
                    tmp_box.append(iou_2d)

                good_k = range_h[np.argmax(np.array(tmp_box))] + val1[0]
                pred[j, 12] = pred[j, 12].astype(np.float32) - pred[j, 8].astype(np.float32) / 2 + good_k / 2
                pred[j, 8] = good_k


            np.savetxt(os.path.join(target_dir, i), pred, fmt='%s')




def eval_from_scrach(gt_dir, det_dir, ap_mode=40):
    global AP_mode
    AP_mode = ap_mode

    # det_dir += '/data'

    # gt_file = np.loadtxt('/private/personal/pengliang/KITTI_split/val.txt', dtype=str)

    all_gt, all_det = [], []
    all_f = sorted(os.listdir(det_dir))
    for i, f in enumerate(tqdm(all_f)):
        gt_f = np.loadtxt(os.path.join(gt_dir, f), dtype=str).reshape(-1, 15)
        # gt_f = np.loadtxt(os.path.join(gt_dir, gt_file[i]+'.txt'), dtype=str).reshape(-1, 15)
        det_f = np.loadtxt(os.path.join(det_dir, f), dtype=str).reshape(-1, 16)

        # if len(det_f) > 0:
        #     det_f[:, -2] = 1.57
        # if len(gt_f) > 0:
        #     gt_f[:, -1] = 1.57

        # if len(det_f) > 0:
        #     # det_f[:, -1] = det_f[:, -1].astype(np.float32) * np.exp(-det_f[:, 13].astype(np.float32) / 40.)
        #     # det_f = refine_score_2d_3d(os.path.join(det_dir, f), '/private/personal/pengliang/KITTI3D/testing/calib')
        #     det_f = refine_score_2d_3d(os.path.join(det_dir, f), '/pvc_data/personal/pengliang/KITTI3D/testing/calib')


        # # # det_f = refine_score_2d_3d(os.path.join(det_dir, f))
        # # # np.savetxt('/private/personal/pengliang/history_submit/DID-M3D/DID-M3D_v5/data/'+f,
        # # np.savetxt('/private/personal/pengliang/history_submit/DID-M3D/DID-M3D_v6/data/'+f,
        # # np.savetxt('/private/personal/pengliang/history_submit/DID-M3D/DID-M3D_v7/data/'+f,
        # #            det_f, fmt='%s')
        # np.savetxt('/pvc_user/pengliang/history_submit2/TempM3D/flow2/data/'+f,
        #            det_f, fmt='%s')


        gt = {}
        det = {}
        '''bbox'''
        gt['bbox'] = gt_f[:, 4:8].astype(np.float32)
        det['bbox'] = det_f[:, 4:8].astype(np.float32)

        '''alpha'''
        gt['alpha'] = gt_f[:, 3].astype(np.float32)
        det['alpha'] = det_f[:, 3].astype(np.float32)
        # gt['alpha'] = np.abs(gt_f[:, -1].astype(np.float32) % np.pi)
        # det['alpha'] = np.abs(det_f[:, -2].astype(np.float32) % np.pi)

        '''occluded'''
        gt['occluded'] = gt_f[:, 2].astype(np.float32)
        det['occluded'] = det_f[:, 2].astype(np.float32)

        '''truncated'''
        gt['truncated'] = gt_f[:, 1].astype(np.float32)
        det['truncated'] = det_f[:, 1].astype(np.float32)

        '''name'''
        gt['name'] = gt_f[:, 0]
        det['name'] = det_f[:, 0]

        '''location'''
        gt['location'] = gt_f[:, 11:14].astype(np.float32)
        det['location'] = det_f[:, 11:14].astype(np.float32)

        '''dimensions, convert hwl to lhw'''
        gt['dimensions'] = gt_f[:, [10, 8, 9]].astype(np.float32)
        det['dimensions'] = det_f[:, [10, 8, 9]].astype(np.float32)
        # gt['dimensions'] = np.tile(np.array([[4, 1.6, 1.6]]), (len(gt_f), 1))
        # det['dimensions'] = np.tile(np.array([[4, 1.6, 1.6]]), (len(det_f), 1))

        '''rotation_y'''
        gt['rotation_y'] = gt_f[:, 14].astype(np.float32)
        det['rotation_y'] = det_f[:, 14].astype(np.float32)

        '''score'''
        det['score'] = det_f[:, 15].astype(np.float32)
        # det['score'] = np.ones_like(det_f[:, 14].astype(np.float32))

        ''' append to tail'''
        all_gt.append(gt)
        all_det.append(det)

    if AP_mode == 40:
        print('-' * 40 + 'AP40 evaluation' + '-' * 40)
    if AP_mode == 11:
        print('-' * 40 + 'AP11 evaluation' + '-' * 40)

    # print('------------------evalute model: {}--------------------'.format(det_dir.split('/')[-3]))
    print('------------------evalute model: {}--------------------'.format(det_dir.split('/')[-2]))
    for cls in ['Car', 'Pedestrian', 'Cyclist']:
    # for cls in ['Car']:
        # for cls in ['Car']:
        print('*' * 20 + cls + '*' * 20)
        res = get_official_eval_result(all_gt, all_det, cls, z_axis=1, z_center=1)
        Car_res = res['detail'][cls]
        for k in Car_res.keys():
            print(k, Car_res[k])
    print('\n')

if __name__ == '__main__':
    # # gt_dir = '/private/pengliang/KITTI3D/training/label_2'


    # import cv2 as cv
    # depth_dir = '/pvc_data/personal/pengliang/KITTI3D/training/lidar_depth/'
    # split = np.loadtxt('/pvc_data/personal/pengliang/KITTI3D/ImageSets/train.txt', dtype=str)
    # pre_d = np.zeros((384, 1280))
    # cnt_d = np.zeros((384, 1280))
    # for v in tqdm(split):
    #     f = os.path.join(depth_dir, v+'.png')
    #     cur_d = cv.imread(f, -1) / 256.
    #     cur_d = cv.resize(cur_d, (1280, 384), interpolation=cv.INTER_NEAREST)
    #
    #     mask_common = (pre_d > 0) & (cur_d > 0)
    #     pre_d[mask_common] += cur_d[mask_common]
    #     cnt_d[mask_common] += 1
    #
    #     mask_new = (pre_d == 0) & (cur_d > 0)
    #     pre_d[mask_new] = cur_d[mask_new]
    #     cnt_d[mask_new] += 1
    #
    # final_depth = pre_d / cnt_d
    # final_depth[final_depth != final_depth] = 0
    # c = 1



    # eval_from_scrach('/private/personal/pengliang/KITTI3D/training/label_2',
    #                  # '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_rep_0521/EPOCH_200/data', ap_mode=40)
    #                  '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_ablation_full_rep_1/EPOCH_140/data', ap_mode=40)
    # exit(0)
    #
    # eval_from_scrach('/private/personal/pengliang/OpenPCDet/pred/data',
    #                  '/pvc_user/pengliang/MonoCon/mmdetection3d-0.14.0/outputs/pkl2txt', ap_mode=40)
    # exit(0)

    # gt_dir = '/pvc_data/personal/pengliang/KITTI3D/training/label_2'
    # gt_dir = '/private/pengliang/OpenPCDet/pred/data'
    # gt_dir = '/private/personal/pengliang/OpenPCDet/pred/data'
    gt_dir = '/pvc_data/personal/pengliang/OpenPCDet/pred/data'
    # # det_dir = '/private/pengliang/GUPNet_master/GUPNet-main/code/outputs/data'
    # # det_dir = '/private/pengliang/GUPNet_master/GUPNet-main/code/outputs_reproduce_v3/EPOCH_140/data'
    # # det_dir = '/private/pengliang/GUPNet_master/GUPNet-main/code/outputs_reproduce_v2/EPOCH_140/data'
    # # det_dir = '/private/pengliang/GUPNet_master/GUPNet-main/code/debug_move_depth/EPOCH_140/data'
    # # for i in range(110, 131, 10):
    # for i in range(140, 141, 10):
    #
    #     # det_dir = '/private/pengliang/GUPNet_master/GUPNet-main/code/outputs_reproduce_abla_base/EPOCH_{}/data'.format(i)
    #     det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern_RoIAlign_InsDepth_Uncern_removeOrgDepth_fixInferDepth_v2_removeGeo_300_LogDepth_1/EPOCH_{}/data'.format(i)
    #     # det_dir = '/private/pengliang/GUPNet_master/GUPNet-main/code/outputs_reproduce_abla_base_nocrop/EPOCH_{}/data'.format(i)
    #     eval_from_scrach(gt_dir, det_dir, ap_mode=40)

    # for i in range(400000, 500000, 10000):
    #     # det_dir = '/private/personal/pengliang/M3D_RPN/output_lidar_det_100_iter_500000/kitti_3d_multi_warmup/results/results_{}/data'.format(i)
    #     # det_dir = '/private/personal/pengliang/M3D_RPN/output_lidar_det_200_iter_500000/kitti_3d_multi_warmup/results/results_{}/data'.format(i)
    #     # det_dir = '/private/personal/pengliang/M3D_RPN/output_lidar_det_500_iter_500000/kitti_3d_multi_warmup/results/results_{}/data'.format(i)
    #     det_dir = '/private/personal/pengliang/M3D_RPN/output_lidar_det_1000_iter_500000/kitti_3d_multi_warmup/results/results_{}/data'.format(i)
    #     eval_from_scrach('/private/personal/pengliang/KITTI3D/training/label_2', det_dir, ap_mode=40)
    # exit(0)


    # # det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_ablation_full_1/EPOCH_140/data'
    # # det_dir = '/private/personal/pengliang/GUPNet_master/GUPNet-main/code/outputs_reproduce_v3/EPOCH_140/data'
    # for i in range(120, 160, 10):
    #     # det_dir = '/private/personal/pengliang/GUPNet_master/GUPNet-main/code/outputs_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern/EPOCH_{}/data'.format(i)
    #     det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_ablation_base/EPOCH_{}/data'.format(i)
    #     # det_dir = '/private/personal/pengliang/GUPNet_master/GUPNet-main/code/outputs_no_conf/checkpoint_epoch_140/data'
    #     eval_from_scrach('/private/personal/pengliang/KITTI3D/training/label_2', det_dir, ap_mode=40)
    # exit(0)

    '''
    ********************Car********************
bbox@0.70 [98.82691049580872, 92.8267722127073, 85.5441109733819]
bev@0.70 [31.101812524261153, 22.764191324948406, 19.50104181431937]
3d@0.70 [22.97913916063991, 16.119232579740604, 14.031035166019754]
aos [97.73858716249266, 91.08132833470174, 83.06192060277077]
bev@0.50 [66.50451129714669, 49.73876377532684, 45.01655175902264]
3d@0.50 [60.089517009783414, 45.48664979806176, 40.96343115832729]
bev@0.30 [86.2729249053198, 70.78553406631798, 65.79441330497954]
3d@0.30 [85.3193582095192, 68.23708531781432, 63.21770935024171]
********************Pedestrian********************
bbox@0.50 [72.82782680827673, 63.54751997615832, 54.31490053271239]
bev@0.50 [7.861879652781592, 6.20408646498005, 5.141301973758383]
3d@0.50 [6.334921872927309, 5.146994095317965, 4.252115390251399]
aos [55.92488408964377, 49.511982383406114, 42.41245119129039]
bev@0.25 [32.27569403624766, 26.971758325857508, 21.74288058271682]
3d@0.25 [31.47075106981069, 25.31439946101163, 21.216644410797098]
********************Cyclist********************
bbox@0.50 [77.1036612237046, 50.552528655599616, 47.71997986142494]
bev@0.50 [5.22978967660947, 2.682367691535422, 2.4999216658845738]
3d@0.50 [4.966435984232765, 2.5739121343957847, 2.39162173735905]
aos [74.57464613257832, 48.447175475314694, 45.6741594858569]
bev@0.25 [21.287716174890786, 11.723145087464037, 11.169385682958872]
3d@0.25 [21.0578035670051, 11.555899899305382, 10.559655380617935]


********************Pedestrian********************
bbox@0.50 [66.65438732588782, 57.18065840369052, 47.62930210565195]
bev@0.50 [13.239303143839166, 9.927903873093202, 8.203762870021828]
3d@0.50 [11.620121395108288, 9.08126244752176, 6.932585064758934]
aos [51.0767182356761, 44.415585987081236, 37.27733803593227]
bev@0.25 [32.4585788741065, 26.82354423874426, 21.400227229032893]
3d@0.25 [32.11931296254194, 25.37511353447804, 21.16425595567784]
********************Cyclist********************
bbox@0.50 [66.87467397543242, 39.765361073220134, 35.5297154370935]
bev@0.50 [7.561936527529281, 4.181650761379926, 3.615552825230984]
3d@0.50 [6.977005619898051, 3.65281166892777, 3.5294497997499024]
aos [64.58257782353243, 37.6460017987607, 33.455482742926854]
bev@0.25 [28.17359859247305, 15.530830315632848, 13.974962894472476]
3d@0.25 [27.47245850010328, 14.297642251218324, 13.625583625409424]

    '''

    # gt_dir = '/private/personal/pengliang/OpenPCDet/pred/data'
    # det_dir = '/private/personal/pengliang/GUPNet_hcx/data'
    gt_dir = '/pvc_data/personal/pengliang/KITTI3D/training/label_2'

    for i in range(130, 180, 10):
        eval_from_scrach(gt_dir, "/pvc_user/chenghaoran/tempM3D/DID/work_dir/TIP/TIP_ori/20231226_211214/outputs_flow_raft/EPOCH_{}/data".format(str(i)), ap_mode=40)
    # for i in range(120, 300, 10):
    # for i in range(220, 300, 10):
    # for i in range(280, 290, 10):
    # for i in range(140, 150, 10):
    # for i in range(130, 140, 10):
    # for i in range(140, 150, 10):
    # for i in range(110, 200, 10):
    # for i in range(140, 150, 10):
    # for i in range(130, 140, 10):
    # for i in range(110, 160, 10):
    # for i in range(21, 31):
    # for i in range(50, 61):
    # for i in range(52, 61):
    # for i in range(55, 76):
    # for i in range(65, 76):
    # for i in range(55, 81):
    # for i in range(51, 81):
        # det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_no_conf3/checkpoint_epoch_{}/data'.format(i)
        # det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_no_conf3_1/checkpoint_epoch_{}/data'.format(i)
        # det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_no_conf3_1/checkpoint_epoch_{}/data'.format(170)
        # det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern_RoIAlign_InsDepth_Uncern_removeOrgDepth_fixInferDepth_v2_removeGeo_300_LogDepth_test_car_2/EPOCH_{}/data'.format(i)
        # det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern_RoIAlign_InsDepth_Uncern_removeOrgDepth_fixInferDepth_v2_removeGeo_300_LogDepth_test_car_newNet_1_1/EPOCH_{}/data'.format(i)
        # det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern_RoIAlign_InsDepth_Uncern_removeOrgDepth_fixInferDepth_v2_removeGeo_300_LogDepth_test_car_newNet2_1/EPOCH_{}/data'.format(i)
        # det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern_RoIAlign_InsDepth_Uncern_removeOrgDepth_fixInferDepth_v2_removeGeo_300_LogDepth_test_car_newNet2_aux_2/EPOCH_{}/data'.format(i)
        # det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern_RoIAlign_InsDepth_Uncern_removeOrgDepth_fixInferDepth_v2_removeGeo_300_LogDepth_test_car_newNet2_aux_fixAug2_1/EPOCH_{}/data'.format(i)
        # det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern_RoIAlign_InsDepth_Uncern_removeOrgDepth_fixInferDepth_v2_removeGeo_300_LogDepth_test_car_newNet2_aux_AdamW/EPOCH_{}/data'.format(i)
        # det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern_RoIAlign_InsDepth_Uncern_removeOrgDepth_fixInferDepth_v2_removeGeo_300_LogDepth_test_car_newNet2_aux_ScaleOffset/EPOCH_{}/data'.format(i)
        # det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern_RoIAlign_InsDepth_Uncern_removeOrgDepth_fixInferDepth_v2_removeGeo_300_LogDepth_test_car_newNet2_aux_2/EPOCH_{}/data'.format(i)
        # det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern_RoIAlign_InsDepth_Uncern_removeOrgDepth_fixInferDepth_v2_removeGeo_300_LogDepth_test_car_newNet2_aux_AdamW/EPOCH_{}/data'.format(i)
        # det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern_RoIAlign_InsDepth_Uncern_removeOrgDepth_fixInferDepth_v2_removeGeo_300_LogDepth_test_car_newNet2_aux_AdamW2/EPOCH_{}/data'.format(i)
        # det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern_RoIAlign_InsDepth_Uncern_removeOrgDepth_fixInferDepth_v2_removeGeo_300_LogDepth_test_car_newNet2_aux_2/EPOCH_{}/data'.format(i)
        # det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_largeScaleShift_EnlargeNet_1/EPOCH_{}/data'.format(i)
        # det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/tools/outputs_tmp2/checkpoint_epoch_150/data'
        # det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_test_tmp/checkpoint_epoch_140/data'
        # det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_vedio_offset_flow_fixFlip_noBk3Frame_UpFusion_warpV4_3/EPOCH_{}/data'.format(i)
        # det_dir = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_ablation_full_rep_1/EPOCH_{}/data'.format(i)
        # det_dir = '/pvc_user/pengliang/DID/DID-main/code/outputs_test_tmp/checkpoint_epoch_160/data'.format(i)
        # det_dir = '/pvc_user/pengliang/DID/DID-main/code/outputs_test_tmp/checkpoint_epoch_140/data'.format(i)
        # det_dir = '/pvc_user/pengliang/DID-M3D/kitti_models/output/DID-M3D/EPOCH_110/data'
        # det_dir = '/pvc_user/pengliang/DID/DID-main/code/outputs_DID_UpDepthLossNet2NoBN_test/EPOCH_{}/data'.format(i)
        # det_dir = '/pvc_user/pengliang/DID-M3D/kitti_models/output/DID-M3D_3/EPOCH_140/data'
        # det_dir = '/pvc_user/pengliang/DID-M3D/tmp_test/test/checkpoint_epoch_130/data'
        # det_dir = '/pvc_user/pengliang/DID/DID-main/code/outputs_DID_TempM3D/EPOCH_{}/data'.format(i)
        # det_dir = '/pvc_user/xujunkai/LIGA-Stereo/outputs/configs_stereo_kitti_models/liga.trainval.test/eval/eval_with_train/epoch_20/test/final_result/data'
        # det_dir = '/pvc_user/xujunkai/LIGA-Stereo/outputs/configs_stereo_kitti_models/liga.trainval.test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/xujunkai/LIGA-Stereo/outputs/configs_stereo_kitti_models/liga.trainval.trainval-60/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/CaDDN_sup3D/CaDDN_3dConvL3NoBN_DLADD3DLowFeat_WeightProb_ProbL3NoBN_OccupancyV4FocalLoss_FixCenterBugW10_test_2/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/CaDDN_sup3D/CaDDN_3dConvL3NoBN_DLADD3DLowFeat_WeightProb_ProbL3NoBN_OccupancyV4FocalLoss_FixCenterBugW10_test_2/eval/eval_with_train/epoch_68/test/final_result/data'
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/CaDDN_sup3D/CaDDN_3dConvL3NoBN_DLADD3DLowFeat_NoDD3D_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/CaDDN_sup3D/CaDDN_3dConvL3NoBN_DLADD3DLowFeat_NoDD3D_test_2/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/CaDDN_sup3D/CaDDN_3dConvL3NoBN_DLADD3DLowFeat_WeightProb_ProbL3NoBN_OccupancyV4FocalLoss_FixCenterBugW10_NoDD3D_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/CaDDN_sup3D/CaDDN_3dConvL3NoBN_DLADD3DLowFeat_WeightProb_ProbL3NoBN_OccupancyV4FocalLoss_FixCenterBugW10_NoDD3D_test_2/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/CaDDN_sup3D/CaDDN_3dConvL3NoBN_DLADD3DLowFeat_WeightProb_ProbL3NoBN_OccupancyV4FocalLoss_FixCenterBugW10_NoDD3D_test_2/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/CaDDN_sup3D/CaDDN_3dConvL3NoBN_DLADD3DLowFeat_WeightProb_ProbL3NoBN_OccupancyV5BilateralFocalLoss_FixCenterBug_test/eval/epoch_61/test/default/final_result/data'
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/CaDDN_sup3D/CaDDN_3dConvL3NoBN_DLADD3DLowFeat_WeightProb_ProbL3NoBN_OccupancyV5BilateralFocalLoss_FixCenterBugW10_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/CaDDN_sup3D/CaDDN_3dConvL3NoBN_DLADD3DLowFeat_WeightProb_ProbL3NoBN_OccupancyV5BilateralFocalLoss_FixCenterBug_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/CaDDN_sup3D/CaDDN_3dConvL3NoBN_DLADD3DLowFeat_WeightProb_ProbL3NoBN_OccupancyV5BilateralFocalLoss_FixCenterBug_NoDD3D_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/CaDDN_sup3D/CaDDN_3dConvL3NoBN_DLADD3DLowFeat_WeightProb_ProbL3NoBN_OccupancyV5BilateralFocalLoss_FixCenterBug_NoDD3D_test_2/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/CaDDN_sup3D/CaDDN_3dConvL3NoBN_DLADD3DLowFeat_WeightProb_ProbL3NoBN_OccupancyV5BilateralBceLoss_FixCenterBug_NetInit_Erode_UpFgWeight_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/CaDDN_sup3D/CaDDN_3dConvL3NoBN_DLADD3DLowFeat_WeightProb_ProbL3NoBN_OccupancyV5BilateralBceLoss_FixCenterBug_NetInit_Erode_UpFgWeight_test_1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/CaDDN_sup3D/CaDDN_3dConvL3NoBN_DLADD3DLowFeat_WeightProb_ProbL3NoBN_OccupancyV5BilateralFocalLoss_FixCenterBug_NetInit_Erode_regBEV_3D2Fru3DBN_FruProb_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_test_1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_test_5/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_test_5/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_test_4_fixFlip/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_test_1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_depthDense_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_depthDense_test_1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_depthDense_b4_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_depthDense_b4_test_1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb3DProb_LossW10_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb3DProb_LossW10_test_1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb3DProb_LossW10_depthDense_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb3DProb_LossW10_depthDense_test_1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb3DProb_LossW10_1209v1_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb3DProb_LossW10_1209v1_test_1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb3DProb_LossW10_1209v2_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb3DProb_LossW10_1209v2_test_1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb3DProb_LossW10_1209v1Norm0.1_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb3DProb_LossW10_1209v1Norm0.1_test_1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb3DProb_LossW10_1209v2Norm0.1_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb3DProb_LossW10_1209v2Norm0.1_test_1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_catDHW_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_catDHW_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2Distort_catDHW_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2Distort_catDHW_AdamW1_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2Distort_fix3D_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2Distort_fix3D_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_fix3D_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_fix3D_test_4/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_fix3D_RefineVoxel_test_2/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_fix3D_LossWeight1_test_2/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_fix3D_LossWeight30_test_1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_fix3D_DD3D_test_1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_fix3D_LossWeight1_HourNoBN_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_fix3D_LossWeight1_HourNoBNFru3D_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_fix3D_LossWeight1_HourNoBNFru3D_test_1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_fix3D_LossWeight1_HourNoBNFruNoBN3D_test_1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_fix3D_LossWeight1_HourNoBNFruNoBN3D_test_2/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_fix3D_LossWeight1_HourNoBN_test_1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_fix3D_LossWeight1_HourNoBNFru_test/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_fix3D_test_1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixFlip_fix3D_test_2/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixFlip_fix3D_test_1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/pengliang/MonoVoxel/OpenPCDet/output/cfgs/kitti_models/Occ/OccAffine/FruProb_fixAffineAugV2_depthDense_test_1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/xujunkai/Monomine/outputs/configs_stereo_kitti_models/liga.3d-and-bev.trainval.pe-cat-left-rgbd-right-rgb/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/xujunkai/Monomine/outputs/configs_stereo_kitti_models/liga.3d-and-bev.trainval.pe-cat-left-rgbd-right-none/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/xujunkai/Monomine/outputs/configs_stereo_kitti_models/liga.3d-and-bev.trainval.sdf-conv3d-layer3-trainval-0/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/xujunkai/Monomine/outputs/configs_stereo_kitti_models/liga.3d-and-bev.trainval.sdf-conv3d-layer3-trainval-1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/xujunkai/MonoNeRD/outputs/configs_stereo_kitti_models/mononerd.3d-and-bev.trainval.new-qkv-left-rgbd-0/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/xujunkai/MonoNeRD/outputs/configs_stereo_kitti_models/mononerd.3d-and-bev.trainval.new-qkv-left-rgbd-sdf-0/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/xujunkai/MonoNeRD/outputs/configs_stereo_kitti_models/mononerd.3d-and-bev.trainval.new-qkv-left-rgb-right-rgb-0/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/xujunkai/MonoNeRD/outputs/configs_stereo_kitti_models/mononerd.3d-and-bev.trainval.qkv-left-rgbd-sdf-right-rgb-0/eval/eval _with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/xujunkai/MonoNeRD/outputs/configs_stereo_kitti_models/mononerd.3d-and-bev.trainval.qkv-left-rgbd-sdf-right-rgb-1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/xujunkai/MonoNeRD/outputs/configs_stereo_kitti_models/mononerd.3d-and-bev.trainval.qkv-left-rgbd-sdf-right-rgbd-0/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/xujunkai/MonoNeRD/outputs/configs_stereo_kitti_models/mononerd.3d-and-bev.trainval.qkv-left-rgbd-sdf-right-rgbd-1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/xujunkai/MonoNeRD/outputs/configs_stereo_kitti_models/mononerd.3d-and-bev.trainval.new-qkv-left-rgbd-sdf-right-rgb-0/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/xujunkai/MonoNeRD/outputs/configs_stereo_kitti_models/mononerd.3d-and-bev.trainval.new-qkv-left-rgbd-sdf-right-rgb-1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/xujunkai/MonoNeRD/outputs/configs_stereo_kitti_models/mononerd.3d-and-bev.trainval.ssim-qkv-left-rgbd-sdf-right-rgb-0/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/xujunkai/MonoNeRD/outputs/configs_stereo_kitti_models/mononerd.3d-and-bev.trainval.ssim-qkv-left-rgbd-sdf-right-rgb-1/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/xujunkai/MonoNeRD/outputs/configs_stereo_kitti_models/mononerd.3d-and-bev.trainval.qkv-left-rgb1d1-sdf10-right-rgb5-2/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/pvc_user/xujunkai/MonoNeRD/outputs/configs_stereo_kitti_models/mononerd.3d-and-bev.trainval.qkv-left-rgbd-sdf-right-rgbd-noimitation-0/eval/eval_with_train/epoch_{}/test/final_result/data'.format(i)
        # det_dir = '/private/personal/pengliang/history_submit/history_submit_m3d/LPCG_monoflex/data/'

        # det_dir = '/private/personal/pengliang/history_submit/DID-M3D/original_dig/data'

        # post_3d('/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/outputs_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern_RoIAlign_InsDepth_Uncern_removeOrgDepth_fixInferDepth_v2_removeGeo_300_LogDepth_test_car_newNet_1_1/EPOCH_{}/data'.format(i),
        #         '/private/personal/pengliang/history_submit/DID-M3D/original_dig/data')
        # print(i)
        # eval_from_scrach(gt_dir, det_dir, ap_mode=40)
        # eval_from_scrach(gt_dir, '/private/personal/pengliang/history_submit/DID-M3D/DID-M3D_v3/data', ap_mode=40)
        # eval_from_scrach(gt_dir, '/private/personal/pengliang/history_submit/DID-M3D/DID-M3D_v4/data', ap_mode=40)


'''
FruProb_fixAffineAugV2_fix3D_LossWeight1_HourNoBN_test 
68
68
100%|██████████████████████████████████████| 7518/7518 [00:37<00:00, 197.99it/s]
----------------------------------------AP40 evaluation----------------------------------------
------------------evalute model: final_result--------------------
********************Car********************
bbox@0.70 [76.60456803713429, 72.0875879646869, 72.0875879646869]
bev@0.70 [28.374671157838723, 19.052232096947105, 19.052232096947105]
3d@0.70 [22.535899519928858, 15.574688640485848, 15.574688640485848]
aos [31.52399397700747, 30.12037669418784, 30.12037669418784]
bev@0.50 [51.6469554440259, 38.98188036672925, 38.98188036672925]
3d@0.50 [48.804138938274306, 36.32577327864104, 36.32577327864104]
bev@0.30 [66.17594051239496, 55.15206276140317, 55.15206276140317]
3d@0.30 [65.57001086567139, 53.07052003096667, 53.07052003096667]

Car (Detection)	95.78 %	92.29 %	84.96 %
Car (Orientation)	95.71 %	92.05 %	84.63 %
Car (3D Detection)	25.55 %	17.02 %	14.79 %
Car (Bird's Eye View)	35.38 %	24.18 %	21.37 %
Pedestrian (Detection)	57.08 %	42.95 %	38.28 %
Pedestrian (Orientation)	40.08 %	29.19 %	25.79 %
Pedestrian (3D Detection)	14.68 %	9.15 %	7.80 %
Pedestrian (Bird's Eye View)	16.54 %	10.65 %	9.16 %
Cyclist (Detection)	41.76 %	30.22 %	26.32 %
Cyclist (Orientation)	25.39 %	17.73 %	14.71 %
Cyclist (3D Detection)	7.37 %	3.56 %	2.84 %
Cyclist (Bird's Eye View)	8.58 %	4.35 %	3.55 



70
100%|██████████████████████████████████████| 7518/7518 [00:37<00:00, 198.30it/s]
----------------------------------------AP40 evaluation----------------------------------------
------------------evalute model: final_result--------------------
********************Car********************
bbox@0.70 [76.69076278885683, 72.1383761324854, 72.1383761324854]
bev@0.70 [27.269087149237812, 18.981998196642813, 18.981998196642813]
3d@0.70 [23.218177717129457, 15.729758271564249, 15.729758271564249]
aos [31.49227479445932, 30.123997104977057, 30.123997104977057]
bev@0.50 [51.61751682547545, 39.072156986612875, 39.072156986612875]
3d@0.50 [48.6613501808237, 36.36366784079164, 36.36366784079164]
bev@0.30 [66.08639853782495, 55.17670022571466, 55.17670022571466]
3d@0.30 [64.03893062387898, 53.1136806027773, 53.1136806027773]

Car (Detection)	95.78 %	92.29 %	84.96 %
Car (Orientation)	95.70 %	92.05 %	84.63 %
Car (3D Detection)	24.74 %	16.70 %	14.60 %
Car (Bird's Eye View)	34.51 %	23.74 %	21.18 %
Pedestrian (Detection)	56.38 %	41.60 %	37.62 %
Pedestrian (Orientation)	39.55 %	28.42 %	25.36 %
Pedestrian (3D Detection)	14.66 %	9.16 %	7.81 %
Pedestrian (Bird's Eye View)	16.35 %	10.58 %	8.98 %
Cyclist (Detection)	43.60 %	32.08 %	27.68 %
Cyclist (Orientation)	25.06 %	17.98 %	14.98 %
Cyclist (3D Detection)	8.41 %	4.25 %	3.61 %
Cyclist (Bird's Eye View)	9.54 %	5.27 %	4.09 %



FruProb_fixAffineAugV2_fix3D_LossWeight1_HourNoBNFruNoBN3D_test_2
70
100%|███████████████████████████████████████| 7518/7518 [05:47<00:00, 21.61it/s]
----------------------------------------AP40 evaluation----------------------------------------
------------------evalute model: final_result--------------------
********************Car********************
bbox@0.70 [76.78416892782313, 72.19256156187728, 72.19256156187728]
bev@0.70 [26.91559156563595, 18.91802627148343, 18.91802627148343]
3d@0.70 [22.62003840724142, 15.99572453571853, 15.99572453571853]
aos [31.765422148134654, 30.291680435405553, 30.291680435405553]
bev@0.50 [49.79200155460347, 38.677616492118595, 38.677616492118595]
3d@0.50 [47.220122940976935, 36.2798745253479, 36.2798745253479]
bev@0.30 [65.77305113352173, 54.85915758866617, 54.85915758866617]
3d@0.30 [63.72606057078719, 52.7690574019697, 52.7690574019697]



FruProb_fixAffineAugV2_fix3D_test occm3d
68
100%|██████████████████████████████████████| 7518/7518 [00:30<00:00, 242.84it/s]
----------------------------------------AP40 evaluation----------------------------------------
------------------evalute model: final_result--------------------
********************Car********************
bbox@0.70 [76.43979884094236, 71.74428869683953, 71.74428869683953]
bev@0.70 [28.54501903875788, 19.24786309404947, 19.24786309404947]
3d@0.70 [22.867199801375744, 15.269180562787824, 15.269180562787824]
aos [31.74112257696038, 30.24719194341988, 30.24719194341988]
bev@0.50 [51.36098126475176, 39.01828502171768, 39.01828502171768]
3d@0.50 [48.280759582442485, 36.1902410348302, 36.1902410348302]
bev@0.30 [66.1591323262725, 55.324021600074836, 55.324021600074836]
3d@0.30 [64.2488679857147, 54.52769130841053, 54.52769130841053]
********************Pedestrian********************

'''

'''

FruProb_fixAffineAugV2_depthDense_test 61
100%|██████████████████████████████████████| 7518/7518 [00:31<00:00, 241.49it/s]
----------------------------------------AP40 evaluation----------------------------------------
------------------evalute model: final_result--------------------
********************Car********************
bbox@0.70 [77.05467060644372, 72.30919167199983, 72.30919167199983]
bev@0.70 [28.133014841027492, 18.926203750904598, 18.926203750904598]
3d@0.70 [22.582864058817744, 15.726649314820756, 15.726649314820756]
aos [32.03222227726483, 30.441658914047988, 30.441658914047988]
bev@0.50 [51.20330640776485, 38.675085511019205, 38.675085511019205]
3d@0.50 [48.33944520168008, 36.046759791280216, 36.046759791280216]
bev@0.30 [66.41793468893187, 56.50523944136052, 56.50523944136052]
3d@0.30 [65.84812459861757, 54.45738820210507, 54.45738820210507]
********************Pedestrian********************
'''


'''
flow2 with post_conf
bbox@0.70 [82.89066328909345, 74.93783716914508, 74.93783716914508]
bev@0.70 [29.440381010818683, 19.462530012897556, 19.462530012897556]
3d@0.70 [24.35767420317579, 16.196042444401662, 16.196042444401662]
aos [33.78649648827281, 30.928983124658764, 30.928983124658764]
bev@0.50 [52.64955924209591, 39.133940829650406, 39.133940829650406]
3d@0.50 [49.543671340299305, 36.343275485702286, 36.343275485702286]
bev@0.30 [67.87281019843981, 54.010770752327474, 54.010770752327474]
3d@0.30 [67.3487895302843, 53.423614059708925, 53.423614059708925]

'''

'''
------------------evalute model: GUPNet_hcx--------------------
********************Car********************
bbox@0.70 [79.61759677608103, 67.08947794214446, 67.08947794214446]
bev@0.70 [27.246069961806413, 18.40015186054363, 18.40015186054363]
3d@0.70 [22.129393378793388, 14.333479603043825, 14.333479603043825]
aos [77.37020356789353, 64.82909569877928, 64.82909569877928]
bev@0.50 [49.37460679218697, 36.167926490432365, 36.167926490432365]
3d@0.50 [46.093302522892856, 33.24045637347391, 33.24045637347391]
bev@0.30 [64.86272288939678, 50.99335089635206, 50.99335089635206]
3d@0.30 [64.44253016768037, 48.73407941669428, 48.73407941669428]


bev@0.70 [26.858399986998794, 18.540033471755184, 18.540033471755184]
3d@0.70 [21.980196814644906, 15.296367216079956, 15.296367216079956]

0.3
bbox@0.70 [80.73422823017837, 72.91217464083215, 72.91217464083215]
bev@0.70 [26.910078435782168, 18.594134393207725, 18.594134393207725]
3d@0.70 [22.02491895751523, 15.343321851289907, 15.343321851289907]

0.4
bbox@0.70 [78.48884514203127, 70.69447034722624, 70.69447034722624]
bev@0.70 [26.94819655421738, 18.63883837025565, 18.63883837025565]
3d@0.70 [22.060793918317316, 14.608569632624802, 14.608569632624802]

0.05
bbox@0.70 [84.36669999410539, 78.63200270967869, 78.63200270967869]
bev@0.70 [26.67663574205284, 19.007902526569538, 19.007902526569538]
3d@0.70 [21.808480164535087, 15.135913921944349, 15.135913921944349]
aos [82.29400626540935, 76.35609751427486, 76.35609751427486]

0.1
bbox@0.70 [82.6692991931802, 77.00098153782437, 77.00098153782437]
bev@0.70 [26.766167653379995, 18.456738242425033, 18.456738242425033]
3d@0.70 [21.900041931500894, 15.226168611429557, 15.226168611429557]

------------------evalute model: EPOCH_130--------------------
********************Car********************
bbox@0.70 [83.16871760066743, 75.20945132249054, 75.20945132249054]
bev@0.70 [27.47278595487922, 18.804715609986182, 18.804715609986182]
3d@0.70 [22.54303931076122, 15.600262050660636, 15.600262050660636]
aos [81.04026853160498, 72.93823770108885, 72.93823770108885]
bev@0.50 [51.81391745144506, 38.543786510745385, 38.543786510745385]
3d@0.50 [48.675984728515566, 35.70582830419294, 35.70582830419294]
bev@0.30 [67.87383310026227, 55.79201524293336, 55.79201524293336]
3d@0.30 [65.46514940107664, 53.49479106636892, 53.49479106636892]


------------------evalute model: --------------------
bbox@0.70 [56.16125887332283, 45.99963508370669, 45.99963508370669]
bev@0.70 [29.729753034734674, 20.337810129221054, 20.337810129221054]
3d@0.70 [23.325393097784957, 15.928558371694807, 15.928558371694807]
aos [22.398732447211806, 18.59129403545185, 18.59129403545185]
bev@0.50 [44.43096549816591, 31.977667937019955, 31.977667937019955]
3d@0.50 [42.40164845139433, 30.36325409297464, 30.36325409297464]


---my only car
bbox@0.70 [82.37693769521425, 74.64070749396774, 74.64070749396774]
bev@0.70 [26.69860077071918, 18.04738137896466, 18.04738137896466]
3d@0.70 [20.269476383668287, 13.501165060494166, 13.501165060494166]
aos [80.41055636702362, 72.54721325835106, 72.54721325835106]
bev@0.50 [51.228817541552885, 38.011966020483676, 38.011966020483676]
3d@0.50 [46.11009650442742, 34.851703445630946, 34.851703445630946]
bev@0.30 [65.19471826791606, 53.27386855688786, 53.27386855688786]
3d@0.30 [64.71659904963415, 52.69953013795862, 52.69953013795862]


------------------evalute model: checkpoint_epoch_290--------------------
********************Car********************
bbox@0.70 [82.04580976880808, 74.33770381933856, 74.33770381933856]
bev@0.70 [24.075955595286253, 16.102373891098313, 16.102373891098313]
3d@0.70 [17.8940675564284, 11.841079132317425, 11.841079132317425]
aos [79.98361240334606, 72.14966516035479, 72.14966516035479]
bev@0.50 [48.91938980594589, 37.08213272755261, 37.08213272755261]
3d@0.50 [44.958013327926196, 32.39843551138442, 32.39843551138442]
bev@0.30 [64.8109177718624, 53.05266549093892, 53.05266549093892]
3d@0.30 [64.13546369681379, 52.21340469898903, 52.21340469898903]


------------------evalute model: checkpoint_epoch_170--------------------
********************Car********************
bbox@0.70 [80.04223962045813, 74.45188375538906, 74.45188375538906]
bev@0.70 [24.58155603755085, 16.487831814870972, 16.487831814870972]
3d@0.70 [18.399511152645523, 12.251631618468178, 12.251631618468178]
aos [78.23065979912019, 72.36553294184812, 72.36553294184812]
bev@0.50 [48.98445063619287, 37.06498161050548, 37.06498161050548]
3d@0.50 [45.41587074888304, 32.784065131275334, 32.784065131275334]
bev@0.30 [64.69158682209992, 52.65054266153258, 52.65054266153258]
3d@0.30 [64.07627572136822, 50.356522568963655, 50.356522568963655]


Car (Detection)	96.20 %	90.83 %	80.94 %
Car (Orientation)	96.01 %	90.35 %	80.27 %
Car (3D Detection)	22.00 %	13.09 %	10.93 %
Car (Bird's Eye View)	31.87 %	19.96 %	17.28 
'''


'''
ORIGINAL

你好，这是我昨天又跑了三组的结果，结果如下：
BEV: 29.59, 21.00, 18.71
3D: 21.63, 15.43, 13.02

BEV: 28.79, 20.61, 17.48
3D: 21.51, 15.36, 12.79

BEV: 26.92, 19.87, 16.85
3D: 20.10, 14.08, 12.04

bbox@0.70 [99.01701890026175, 88.66872012951171, 81.21253972179515]
bev@0.70 [29.432278723191168, 21.886702696205912, 18.789002404744032]
3d@0.70 [20.660383194126293, 15.255438332385014, 12.56455014457597]
aos [98.01346436397515, 87.00520265630445, 78.86713682041497]
bev@0.50 [67.42230165534774, 49.65589820058902, 43.34897235346159]
3d@0.50 [61.06490133384982, 44.491169531579445, 39.70826002245665]
bev@0.30 [85.86960433739581, 68.70411779622646, 61.793759630882136]
3d@0.30 [83.30761043606229, 66.21620035315979, 59.2440837755976]

bbox@0.70 [99.1355326594474, 88.85885248025642, 81.34033066178027]
bev@0.70 [30.90470929155765, 22.41418222151852, 19.23684760491952]
3d@0.70 [22.454644792289166, 15.769348241417369, 13.102702468090953]
aos [97.72528884945524, 87.05275940704075, 78.83138222096046]
bev@0.50 [63.807430588727264, 48.74520739047997, 42.729963485137574]
3d@0.50 [58.1233212528433, 43.23438391815787, 38.764648866999565]
bev@0.30 [85.39523570382809, 68.59777419744067, 61.68401297096473]
3d@0.30 [82.32516883247307, 65.71215041247656, 58.88743184819274]


'''

'''
outputs_surface_min_b_8_signle_card_ada

bbox@0.70 [98.23853805205385, 92.44774044204067, 85.15590607571808]
bev@0.70 [31.437573952028984, 22.235583438307753, 18.928103508319495]
3d@0.70 [22.686871794368795, 15.802366796773818, 13.090678067960146]
aos [97.23127023521977, 90.59753951036164, 82.38762911875001]
bev@0.50 [65.17238746776324, 48.556416828789814, 43.538704718595525]
3d@0.50 [59.37519136884355, 44.41779991110735, 38.48172691236825]
bev@0.30 [83.20489194768271, 68.65103095699612, 61.92131414504304]
3d@0.30 [80.95514440819208, 66.20861167338721, 59.4914591413888]
'''



'''
----------------------------------------AP40 evaluation----------------------------------------                                                                                                                                            │··
------------------evalute model: EPOCH_140--------------------                                                                                                                                                                             │··
********************Car********************                                                                                                                                                                                                │··
bbox@0.70 [96.81113561184134, 88.82796294936227, 81.3506221696013]                                                                                                                                                                         │··
bev@0.70 [30.859101553215808, 21.572411148712316, 19.147408340566603]                                                                                                                                                                      │··
3d@0.70 [22.5961620751018, 15.873656085422105, 13.217440276589492]                                                                                                                                                                         │··
aos [96.04057969368658, 87.20786529937938, 78.82603499394871]                                                                                                                                                                              │··
bev@0.50 [65.82844501845607, 49.86349600368568, 43.61505501466571]                                                                                                                                                                         │··
3d@0.50 [61.266574168900526, 44.794303041097, 39.97924246776549]                                                                                                                                                                           │··
bev@0.30 [83.84200068016374, 69.03660027765113, 62.04118734093249]                                                                                                                                                                         │··
3d@0.30 [83.0699595658808, 66.51949822829677, 59.521336634606904]                                                                                                                                                                          │··
                                                                                                                                                                                                                                           │··                                                                                                                                                                                                                         │··
2021-12-24 06:22:52,104   INFO  ==> Saving to checkpoint 'log_reproduce_abla/checkpoints/checkpoint_epoch_140'     


remove depth score

------------------evalute model: checkpoint_epoch_140--------------------                                                                                                                                                                  │··
********************Car********************                                                                                                                                                                                                │··
bbox@0.70 [98.59746352244284, 92.76561464386799, 85.23743611156496]                                                                                                                                                                        │··
bev@0.70 [25.289955846862462, 19.931467632121475, 17.4155276774957]                                                                                                                                                                        │··
3d@0.70 [17.47179827821886, 13.55588383231231, 11.523485498620651]                                                                                                                                                                         │··
aos [97.60387028664648, 91.00989132851076, 82.83220045680618]                                                                                                                                                                              │··
bev@0.50 [62.65739146270663, 47.36805397759398, 42.657027957840995]                                                                                                                                                                        │··
3d@0.50 [57.36215945964859, 42.25238730429484, 37.825172933137516]                                                                                                                                                                         │··
bev@0.30 [83.38836698317269, 67.13175984243408, 61.89777453192544]                                                                                                                                                                         │··
3d@0.30 [82.08389453706675, 65.97464925360732, 59.26887697850294]                                                                                                                                                                          │··
                                                                   


no uncerntainty 

----------------------------------------AP40 evaluation----------------------------------------                                                                                                                                            │··
------------------evalute model: EPOCH_140--------------------                                                                                                                                                                             │··
********************Car********************                                                                                                                                                                                                │··
bbox@0.70 [98.59575044343514, 93.02830857835858, 85.44800134353876]                                                                                                                                                                        │··
bev@0.70 [27.44906217242352, 20.61274100828832, 17.816794731036424]                                                                                                                                                                        │··
3d@0.70 [19.951633079207625, 14.244129987186675, 12.67617156205051]                                                                                                                                                                        │··
aos [97.79247712278902, 91.42911367769052, 83.3238891790053]                                                                                                                                                                               │··
bev@0.50 [64.1871739295141, 49.80900874567687, 43.71179195791306]                                                                                                                                                                          │··
3d@0.50 [58.414990922606044, 44.21893074887075, 39.65591183304065]                                                                                                                                                                         │··
bev@0.30 [83.84208765315991, 69.63745644636936, 64.38312217677726]                                                                                                                                                                         │··
3d@0.30 [82.64879357926253, 67.0026553349696, 61.69899333971806]                                                                                                                                                                           │··
                                                                      
'''


'''
my exp

with NOC depth
----------------------------------------AP40 evaluation----------------------------------------
------------------evalute model: EPOCH_130--------------------
********************Car********************
bbox@0.70 [98.43490571226565, 93.28369604849094, 85.66875873519902]
bev@0.70 [26.691891123530887, 20.412693356561913, 17.510114867508978]
3d@0.70 [18.98882147314751, 14.013668206961386, 11.607911963212569]
aos [97.72547554329215, 91.70665086672933, 83.29952812937131]
bev@0.50 [64.18743410682883, 49.31847872294578, 43.322150952124865]
3d@0.50 [57.82897345713862, 44.22232414246597, 39.57489048613674]
bev@0.30 [85.90816742702484, 69.84208674229507, 64.48005382846388]
3d@0.30 [83.07021180880619, 68.71820917514272, 61.91381165814656]

2021-12-30 19:45:58,963   INFO  ==> Saving to checkpoint 'log_reproduce_abla_base_noc_fix_depthBug_weighted_loss/checkpoints/checkpoint_epoch_130'


with NOC depth uncern

100%|___________________________________________________________________________| 3769/3769 [00:15<00:00, 237.15it/s]
----------------------------------------AP40 evaluation----------------------------------------
------------------evalute model: EPOCH_140--------------------
********************Car********************
bbox@0.70 [98.49718271012013, 93.15447004907963, 85.57197966650783]
bev@0.70 [27.94719673487835, 20.827566479533157, 18.025044316830037]
3d@0.70 [20.11686120689245, 14.187359375623714, 12.346169079750576]
aos [97.47632634861132, 91.22155570689546, 83.0005314470507]
bev@0.50 [60.83630325141638, 46.75150608108486, 42.321579954846094]
3d@0.50 [56.16456503572178, 43.01110885696243, 38.69163393637719]
bev@0.30 [84.73401883137748, 69.0528626100365, 63.89471286209882]
3d@0.30 [81.76182188867276, 66.24417606966296, 61.13949879040822]

2021-12-31 03:21:27,620   INFO  ==> Saving to checkpoint 'log_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern/checkpoints/checkpoint_epoch_140'


with NOC depth uncern and weighted by mean two stage uncern


********************Car********************
bbox@0.70 [98.74329071194133, 92.38632665416606, 85.13003094126361]
bev@0.70 [34.77160545271561, 23.093962085596036, 19.448307901143806]
3d@0.70 [26.146692119522573, 16.138965026344593, 13.634380909532057]
aos [98.14181390329875, 90.93745406549145, 82.87541205032103]
bev@0.50 [68.1330423992115, 49.50049497439816, 44.42174315857808]
3d@0.50 [63.431506688457326, 45.832756578133086, 40.69877462296703]
bev@0.30 [88.5772283740581, 70.72532476419006, 65.38662425619506]
3d@0.30 [86.06431727626725, 68.10855424218605, 62.89441307573169]

'log_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern/checkpoints/checkpoint_epoch_140'

'''


'''
----------------------------------------AP40 evaluation----------------------------------------                                                                                                                                            │··
------------------evalute model: EPOCH_140--------------------                                                                                                                                                                             │··
********************Car********************                                                                                                                                                                                                │··
bbox@0.70 [98.47953523919465, 92.93899074591071, 85.45135636011679]                                                                                                                                                                        │··
bev@0.70 [28.776936712131075, 20.969387132055566, 18.114649437835435]                                                                                                                                                                      │··
3d@0.70 [21.71529671019372, 14.800818033155815, 12.43240009870776]                                                                                                                                                                         │··
aos [97.83515488217404, 91.50362074816523, 83.39937578077227]                                                                                                                                                                              │··
bev@0.50 [65.51865270391079, 48.788495358433245, 43.87498676172596]                                                                                                                                                                        │··
3d@0.50 [60.70210718754456, 44.79551618866028, 38.842968901871735]                                                                                                                                                                         │··
bev@0.30 [86.86852821158398, 70.27655646100747, 64.92031411053135]                                                                                                                                                                         │··
3d@0.30 [84.16043944331847, 67.51855652997253, 62.2003039122472]                                                                                                                                                                           │··
                                                                                                                                                                                                                                           │··
                                                                                                                                                                                                                                           │··
2021-12-31 23:03:14,830   INFO  ==> Saving to checkpoint 'log_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern_RoIAlign_gpu20_1/checkpoints/checkpoint_epoch_140'   
'''

