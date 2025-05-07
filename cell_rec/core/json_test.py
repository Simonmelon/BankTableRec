import argparse
import shutil
import time
from collections import OrderedDict
import json
import sys
import cv2
import torch
from torchvision import transforms
from fitz import Rect
import pandas as pd
# matplotlib.use('TkAgg')
import core.postprocess as postprocess
from core.processor import diaoyong
# sys.path.append("../detr")
# import detr.models
from detr.models import build_model
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MaxResize(object):
    def __init__(self, max_size=1000):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))

        return resized_image


detection_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_class_map(data_type):
    if data_type == 'structure':
        class_map = {
            'table': 0,
            'table column': 1,
            'table row': 2,
            'table column header': 3,
            'table projected row header': 4,
            'table spanning cell': 5,
            'no object': 6
        }
        # class_map = {
        #     'table': 0,
        #     'table cell': 1,
        #     'no object': 2
        # }
    elif data_type == 'detection':
        class_map = {'table': 0, 'table rotated': 1, 'no object': 2}
    return class_map


detection_class_thresholds = {
    "table": 0.5,
    "table rotated": 0.5,
    "no object": 10
}

structure_class_thresholds = {
    "table": 0.5,
    "table column": 0.5,
    "table row": 0.5,
    "table column header": 0.5,
    "table projected row header": 0.5,
    "table spanning cell": 0.5,
    "no object": 10
}

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def preprocess_extract_red_gray(img_path):
    """
    预处理: 提取红色通道并返回灰度 PIL 图像对象
    """
    # 以 BGR 模式读取
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"[WARN] 无法读取图像: {img_path}")
        return None

    # 获取红色通道（二维灰度数组）
    red_channel = image[:, :, 2]

    # 直接将二维数组转换为灰度模式的 PIL 图像
    # Image.fromarray 会自动识别为 "L" 模式（8-bit 灰度）
    gray_red = Image.fromarray(red_channel)

    return gray_red

def iob(bbox1, bbox2):
    """
    Compute the intersection area over box area, for bbox1.
    """
    intersection = Rect(bbox1).intersect(bbox2)

    bbox1_area = Rect(bbox1).get_area()
    if bbox1_area > 0:
        return intersection.get_area() / bbox1_area

    return 0

from copy import deepcopy

def split_and_crop_by_overlap(image, table_rows,
                              overlap_threshold=0.4,
                              output_dir=None):
    """
    返回一个列表，每项 (img_seg, rows, offset)。
    在决定切分前后都做空图/空行检查，避免写入 0×N 的图片。
    """
    tasks = [(image, table_rows, 0)]
    final_segments = []

    while tasks:
        img_seg, rows, offset = tasks.pop(0)
        rows = sorted(rows, key=lambda r: r['bbox'][1])
        did_split = False

        for i in range(len(rows)-1):
            a, b = rows[i], rows[i+1]
            overlap_h = a['bbox'][3] - b['bbox'][1]
            row_h     = a['bbox'][3] - a['bbox'][1]
            if overlap_h > overlap_threshold * row_h:
                cut_y = a['bbox'][1]  # 切线

                # 如果切线在图像边界上，跳过切分
                if cut_y <= 0 or cut_y >= img_seg.height:
                    continue

                # 切成上下两块
                top_img = img_seg.crop((0, 0, img_seg.width,  cut_y))
                bot_img = img_seg.crop((0, cut_y, img_seg.width, img_seg.height))

                # 按 bbox 过滤 rows
                top_rows, bot_rows = [], []
                for r in rows:
                    y1, y2 = r['bbox'][1], r['bbox'][3]
                    if y1 < cut_y:
                        top_rows.append(deepcopy(r))
                    if y2 > cut_y:
                        nr = deepcopy(r)
                        nr['bbox'][1] -= cut_y
                        nr['bbox'][3] -= cut_y
                        bot_rows.append(nr)

                # 只有非空行 & 非零尺寸图才存／加入任务
                if top_rows and top_img.width>0 and top_img.height>0:
                    if output_dir:
                        idx = len(final_segments) + len(tasks)
                        top_img.save(os.path.join(output_dir, f"seg_{idx}_top.jpg"))
                    tasks.insert(0, (top_img, top_rows, offset))

                if bot_rows and bot_img.width>0 and bot_img.height>0:
                    if output_dir:
                        idx = len(final_segments) + len(tasks)
                        bot_img.save(os.path.join(output_dir, f"seg_{idx}_bot.jpg"))
                    # bot 段的 offset 要加上 top_rows 的数量
                    tasks.insert(0, (bot_img, bot_rows, offset + len(top_rows)))

                did_split = True
                break  # 只做一次切分，剩下的留给下一轮

        if not did_split:
            # 本段内无可切分的重叠，加入结果
            final_segments.append((img_seg, rows, offset))

    return final_segments


# def iterative_crop_by_overlap(image, table_rows, output_dir=None,overlap_threshold=0.4, max_iterations=20):
#     # 新增：记录裁剪行号的列表
#     crop_lines = []
#
#     table_rows = sorted(table_rows, key=lambda r: r['bbox'][1])
#     iteration = 0
#     while iteration < max_iterations:
#         overlap_found = False
#         for i in range(len(table_rows) - 1):
#             top_row = table_rows[i]
#             next_row = table_rows[i + 1]
#             row_height = top_row['bbox'][3] - top_row['bbox'][1]
#             if overlap_found:
#                 # 保存裁剪前的中间状态
#                 if output_dir:
#                     temp_path = os.path.join(output_dir,
#                                              f"crop_step_{iteration}_before.jpg")
#                     image.save(temp_path)
#             if next_row['bbox'][1] < top_row['bbox'][3]:
#                 overlap_height = top_row['bbox'][3] - next_row['bbox'][1]
#                 if overlap_height > overlap_threshold * row_height:
#                     # 修改：裁剪线减去1个像素
#                     crop_line = top_row['bbox'][1] - 1
#                     crop_lines.append({
#                         'crop_line': crop_line,
#                         'original_rows': len(table_rows[:i + 1])  # 记录当前裁剪位置之前的行数
#                     })
#
#                     image = image.crop((0, crop_line, image.width, image.height))
#                     # 调整行号时需要累加之前的裁剪偏移
#                     crop_offset = crop_line
#                     for r in table_rows:
#                         r['bbox'][1] -= crop_offset
#                         r['bbox'][3] -= crop_offset
#                     overlap_found = True
#                     break
#             if output_dir:
#                 temp_path = os.path.join(output_dir,
#                                          f"crop_step_{iteration}_after.jpg")
#                 image.save(temp_path)
#         if not overlap_found:
#             break
#         iteration += 1
#     # 返回新增：裁剪记录和调整后的行数
#     if output_dir:
#         final_path = os.path.join(output_dir, "final_cropped.jpg")
#         image.save(final_path)
#     return image, table_rows, crop_lines


def align_headers(headers, rows):
    """
    Adjust the header boundary to be the convex hull of the rows it intersects
    at least 50% of the height of.

    For now, we are not supporting tables with multiple headers, so we need to
    eliminate anything besides the top-most header.
    """

    aligned_headers = []

    for row in rows:
        row['column header'] = False

    header_row_nums = []
    for header in headers:
        for row_num, row in enumerate(rows):
            row_height = row['bbox'][3] - row['bbox'][1]
            min_row_overlap = max(row['bbox'][1], header['bbox'][1])
            max_row_overlap = min(row['bbox'][3], header['bbox'][3])
            overlap_height = max_row_overlap - min_row_overlap
            if overlap_height / row_height >= 0.5:
                header_row_nums.append(row_num)

    if len(header_row_nums) == 0:
        return aligned_headers

    header_rect = Rect()
    if header_row_nums[0] > 0:
        header_row_nums = list(range(header_row_nums[0] + 1)) + header_row_nums

    last_row_num = -1
    for row_num in header_row_nums:
        if row_num == last_row_num + 1:
            row = rows[row_num]
            row['column header'] = True
            header_rect = header_rect.include_rect(row['bbox'])
            last_row_num = row_num
        else:
            # Break as soon as a non-header row is encountered.
            # This ignores any subsequent rows in the table labeled as a header.
            # Having more than 1 header is not supported currently.
            break

    header = {'bbox': list(header_rect)}
    aligned_headers.append(header)

    return aligned_headers


def refine_table_structure(table_structure, class_thresholds):
    """
    Apply operations to the detected table structure objects such as
    thresholding, NMS, and alignment.
    """
    rows = table_structure["rows"]
    columns = table_structure['columns']

    # Process the headers
    column_headers = table_structure['column headers']
    column_headers = postprocess.apply_threshold(column_headers, class_thresholds["table column header"])
    column_headers = postprocess.nms(column_headers)
    column_headers = align_headers(column_headers, rows)

    # Process spanning cells
    spanning_cells = [elem for elem in table_structure['spanning cells'] if not elem['projected row header']]
    projected_row_headers = [elem for elem in table_structure['spanning cells'] if elem['projected row header']]
    spanning_cells = postprocess.apply_threshold(spanning_cells, class_thresholds["table spanning cell"])
    projected_row_headers = postprocess.apply_threshold(projected_row_headers,
                                                        class_thresholds["table projected row header"])
    spanning_cells += projected_row_headers
    # Align before NMS for spanning cells because alignment brings them into agreement
    # with rows and columns first; if spanning cells still overlap after this operation,
    # the threshold for NMS can basically be lowered to just above 0
    spanning_cells = postprocess.align_supercells(spanning_cells, rows, columns)
    spanning_cells = postprocess.nms_supercells(spanning_cells)

    postprocess.header_supercell_tree(spanning_cells)

    table_structure['columns'] = columns
    table_structure['rows'] = rows
    table_structure['spanning cells'] = spanning_cells
    table_structure['column headers'] = column_headers

    return table_structure


def outputs_to_objects(outputs, img_size, class_idx2name):
    m = outputs['pred_logits'].softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = class_idx2name[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects


def objects_to_crops(img, tokens, objects, class_thresholds, padding=10):
    """
    Process the bounding boxes produced by the table detection model into
    cropped table images and cropped tokens.
    """

    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}

        bbox = obj['bbox']
        bbox = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]

        cropped_img = img.crop(bbox)

        table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
        for token in table_tokens:
            token['bbox'] = [token['bbox'][0] - bbox[0],
                             token['bbox'][1] - bbox[1],
                             token['bbox'][2] - bbox[0],
                             token['bbox'][3] - bbox[1]]

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [cropped_img.size[0] - bbox[3] - 1,
                        bbox[0],
                        cropped_img.size[0] - bbox[1] - 1,
                        bbox[2]]
                token['bbox'] = bbox

        cropped_table['image'] = cropped_img
        cropped_table['tokens'] = table_tokens

        table_crops.append(cropped_table)

    return table_crops


def objects_to_structures(objects, tokens, class_thresholds):
    """
    Process the bounding boxes produced by the table structure recognition model into
    a *consistent* set of table structures (rows, columns, spanning cells, headers).
    This entails resolving conflicts/overlaps, and ensuring the boxes meet certain alignment
    conditions (for example: rows should all have the same width, etc.).
    """

    tables = [obj for obj in objects if obj['label'] == 'table']
    table_structures = []

    for table in tables:
        table_objects = [obj for obj in objects if iob(obj['bbox'], table['bbox']) >= 0.5]
        print("objects",objects)
        table_tokens = [token for token in tokens if iob(token['bbox'], table['bbox']) >= 0.5]
        print("table_tokens",table_tokens)
        structure = {}
        for item in objects:
            if item.get('label') == 'table projected row header':
                item['label'] = 'table row'
        columns = [obj for obj in table_objects if obj['label'] == 'table column']
        rows = [obj for obj in table_objects if obj['label'] == 'table row']
        column_headers = [obj for obj in table_objects if obj['label'] == 'table column header']
        spanning_cells = [obj for obj in table_objects if obj['label'] == 'table spanning cell']
        print("spanning_cells",spanning_cells)
        for obj in spanning_cells:
            obj['projected row header'] = True
        projected_row_headers = [obj for obj in table_objects if obj['label'] == 'table projected row header']
        print("projected_row_headers",projected_row_headers)
        for obj in projected_row_headers:
            obj['projected row header'] = True
        spanning_cells += projected_row_headers
        for obj in rows:
            obj['column header'] = False
            for header_obj in column_headers:
                if iob(obj['bbox'], header_obj['bbox']) >= 0.5:
                    obj['column header'] = True

        # Refine table structures
        rows = postprocess.refine_rows(rows, table_tokens, class_thresholds['table row'])
        columns = postprocess.refine_columns(columns, table_tokens, class_thresholds['table column'])

        # Shrink table bbox to just the total height of the rows
        # and the total width of the columns
        row_rect = Rect()
        for obj in rows:
            row_rect.include_rect(obj['bbox'])
        column_rect = Rect()
        for obj in columns:
            column_rect.include_rect(obj['bbox'])
        table['row_column_bbox'] = [column_rect[0], row_rect[1], column_rect[2], row_rect[3]]
        table['bbox'] = table['row_column_bbox']

        # Process the rows and columns into a complete segmented table
        columns = postprocess.align_columns(columns, table['row_column_bbox'])
        rows = postprocess.align_rows(rows, table['row_column_bbox'])

        structure['rows'] = rows
        structure['columns'] = columns
        structure['column headers'] = column_headers
        structure['spanning cells'] = spanning_cells

        if len(rows) > 0 and len(columns) > 1:
            structure = refine_table_structure(structure, class_thresholds)
            print("structure:", structure)
        table_structures.append(structure)

    return table_structures


def structure_to_cells(table_structure, tokens):
    """
    Assuming the row, column, spanning cell, and header bounding boxes have
    been refined into a set of consistent table structures, process these
    table structures into table cells. This is a universal representation
    format for the table, which can later be exported to Pandas or CSV formats.
    Classify the cells as header/access cells or data cells
    based on if they intersect with the header bounding box.
    """
    columns = table_structure['columns']
    rows = table_structure['rows']
    spanning_cells = table_structure['spanning cells']
    cells = []
    subcells = []

    # Identify complete cells and subcells
    for column_num, column in enumerate(columns):
        for row_num, row in enumerate(rows):
            column_rect = Rect(list(column['bbox']))
            row_rect = Rect(list(row['bbox']))
            cell_rect = row_rect.intersect(column_rect)
            header = 'column header' in row and row['column header']
            cell = {'bbox': list(cell_rect), 'column_nums': [column_num], 'row_nums': [row_num],
                    'column header': header}

            cell['subcell'] = False
            for spanning_cell in spanning_cells:
                spanning_cell_rect = Rect(list(spanning_cell['bbox']))
                if (spanning_cell_rect.intersect(cell_rect).get_area()
                    / cell_rect.get_area()) > 0.5:
                    cell['subcell'] = True
                    break

            if cell['subcell']:
                subcells.append(cell)
            else:
                # cell text = extract_text_inside_bbox(table_spans, cell['bbox'])
                # cell['cell text'] = cell text
                cell['projected row header'] = False
                cells.append(cell)

    for spanning_cell in spanning_cells:
        spanning_cell_rect = Rect(list(spanning_cell['bbox']))
        cell_columns = set()
        cell_rows = set()
        cell_rect = None
        header = True
        for subcell in subcells:
            subcell_rect = Rect(list(subcell['bbox']))
            subcell_rect_area = subcell_rect.get_area()
            if (subcell_rect.intersect(spanning_cell_rect).get_area()
                / subcell_rect_area) > 0.5:
                if cell_rect is None:
                    cell_rect = Rect(list(subcell['bbox']))
                else:
                    cell_rect.include_rect(Rect(list(subcell['bbox'])))
                cell_rows = cell_rows.union(set(subcell['row_nums']))
                cell_columns = cell_columns.union(set(subcell['column_nums']))
                # By convention here, all subcells must be classified
                # as header cells for a spanning cell to be classified as a header cell;
                # otherwise, this could lead to a non-rectangular header region
                header = header and 'column header' in subcell and subcell['column header']
        if len(cell_rows) > 0 and len(cell_columns) > 0:
            cell = {'bbox': list(cell_rect), 'column_nums': list(cell_columns), 'row_nums': list(cell_rows),
                    'column header': header, 'projected row header': spanning_cell['projected row header']}
            cells.append(cell)

    # Compute a confidence score based on how well the page tokens
    # slot into the cells reported by the model
    _, _, cell_match_scores = postprocess.slot_into_containers(cells, tokens)
    try:
        mean_match_score = sum(cell_match_scores) / len(cell_match_scores)
        min_match_score = min(cell_match_scores)
        confidence_score = (mean_match_score + min_match_score) / 2
    except:
        confidence_score = 0

    # Dilate rows and columns before final extraction
    # dilated_columns = fill_column_gaps(columns, table_bbox)
    dilated_columns = columns
    # dilated_rows = fill_row_gaps(rows, table_bbox)
    dilated_rows = rows
    for cell in cells:
        column_rect = Rect()
        for column_num in cell['column_nums']:
            column_rect.include_rect(list(dilated_columns[column_num]['bbox']))
        row_rect = Rect()
        for row_num in cell['row_nums']:
            row_rect.include_rect(list(dilated_rows[row_num]['bbox']))
        cell_rect = column_rect.intersect(row_rect)
        cell['bbox'] = list(cell_rect)

    span_nums_by_cell, _, _ = postprocess.slot_into_containers(cells, tokens, overlap_threshold=0.001,
                                                               unique_assignment=True, forced_assignment=False)

    for cell, cell_span_nums in zip(cells, span_nums_by_cell):
        cell_spans = [tokens[num] for num in cell_span_nums]
        # TODO: Refine how text is extracted; should be character-based, not span-based;
        # but need to associate
        cell['cell text'] = postprocess.extract_text_from_spans(cell_spans, remove_integer_superscripts=False)
        cell['spans'] = cell_spans

    # Adjust the row, column, and cell bounding boxes to reflect the extracted text
    num_rows = len(rows)
    rows = postprocess.sort_objects_top_to_bottom(rows)
    num_columns = len(columns)
    columns = postprocess.sort_objects_left_to_right(columns)
    min_y_values_by_row = defaultdict(list)
    max_y_values_by_row = defaultdict(list)
    min_x_values_by_column = defaultdict(list)
    max_x_values_by_column = defaultdict(list)
    for cell in cells:
        min_row = min(cell["row_nums"])
        max_row = max(cell["row_nums"])
        min_column = min(cell["column_nums"])
        max_column = max(cell["column_nums"])
        for span in cell['spans']:
            min_x_values_by_column[min_column].append(span['bbox'][0])
            min_y_values_by_row[min_row].append(span['bbox'][1])
            max_x_values_by_column[max_column].append(span['bbox'][2])
            max_y_values_by_row[max_row].append(span['bbox'][3])
    for row_num, row in enumerate(rows):
        if len(min_x_values_by_column[0]) > 0:
            row['bbox'][0] = min(min_x_values_by_column[0])
        if len(min_y_values_by_row[row_num]) > 0:
            row['bbox'][1] = min(min_y_values_by_row[row_num])
        if len(max_x_values_by_column[num_columns - 1]) > 0:
            row['bbox'][2] = max(max_x_values_by_column[num_columns - 1])
        if len(max_y_values_by_row[row_num]) > 0:
            row['bbox'][3] = max(max_y_values_by_row[row_num])
    for column_num, column in enumerate(columns):
        if len(min_x_values_by_column[column_num]) > 0:
            column['bbox'][0] = min(min_x_values_by_column[column_num])
        if len(min_y_values_by_row[0]) > 0:
            column['bbox'][1] = min(min_y_values_by_row[0])
        if len(max_x_values_by_column[column_num]) > 0:
            column['bbox'][2] = max(max_x_values_by_column[column_num])
        if len(max_y_values_by_row[num_rows - 1]) > 0:
            column['bbox'][3] = max(max_y_values_by_row[num_rows - 1])
    for cell in cells:
        row_rect = Rect()
        column_rect = Rect()
        for row_num in cell['row_nums']:
            row_rect.include_rect(list(rows[row_num]['bbox']))
        for column_num in cell['column_nums']:
            column_rect.include_rect(list(columns[column_num]['bbox']))
        cell_rect = row_rect.intersect(column_rect)
        if cell_rect.get_area() > 0:
            cell['bbox'] = list(cell_rect)
            pass
    print("cells", cells)
    return cells, confidence_score


def cells_to_csv(cells):
    if len(cells) > 0:
        num_columns = max([max(cell['column_nums']) for cell in cells]) + 1
        num_rows = max([max(cell['row_nums']) for cell in cells]) + 1
    else:
        return

    header_cells = [cell for cell in cells if cell['column header']]
    if len(header_cells) > 0:
        max_header_row = max([max(cell['row_nums']) for cell in header_cells])
    else:
        max_header_row = -1

    table_array = np.empty([num_rows, num_columns], dtype="object")
    if len(cells) > 0:
        for cell in cells:
            for row_num in cell['row_nums']:
                for column_num in cell['column_nums']:
                    table_array[row_num, column_num] = cell["cell text"]

    header = table_array[:max_header_row + 1, :]
    flattened_header = []
    for col in header.transpose():
        flattened_header.append(' | '.join(OrderedDict.fromkeys(col)))
    df = pd.DataFrame(table_array[max_header_row + 1:, :], index=None, columns=flattened_header)

    return df.to_csv(index=None)


class TableExtractionPipeline(object):
    def __init__(self, det_device=None, str_device=None,
                 det_model=None, str_model=None,
                 det_model_path='D:\code\\table-transformer-main\TATR-v1.1-All-msft.pth', str_model_path=None,
                 det_config_path=None, str_config_path=None,output_img_dir="processed_images"):

        self.det_device = det_device
        self.str_device = str_device

        self.det_class_name2idx = get_class_map('detection')
        self.det_class_idx2name = {v:k for k, v in self.det_class_name2idx.items()}
        self.det_class_thresholds = detection_class_thresholds
        self.output_img_dir = output_img_dir
        self.str_class_name2idx = get_class_map('structure')
        self.str_class_idx2name = {v:k for k, v in self.str_class_name2idx.items()}
        self.str_class_thresholds = structure_class_thresholds
        os.makedirs(self.output_img_dir, exist_ok=True)

        if not det_config_path is None:
            with open(det_config_path, 'r') as f:
                det_config = json.load(f)
            det_args = type('Args', (object,), det_config)
            det_args.device = det_device
            self.det_model, _, _ = build_model(det_args)
            print("Detection model initialized.")

            if not det_model_path is None:
                self.det_model.load_state_dict(torch.load(det_model_path,
                                                     map_location=torch.device(det_device)))
                self.det_model.to(det_device)
                self.det_model.eval()
                print("Detection model weights loaded.")
            else:
                self.det_model = None

        if not str_config_path is None:
            with open(str_config_path, 'r') as f:
                str_config = json.load(f)
            str_args = type('Args', (object,), str_config)
            str_args.device = str_device
            self.str_model, _, _ = build_model(str_args)
            print("Structure model initialized.")

            if not str_model_path is None:
                self.str_model.load_state_dict(torch.load(str_model_path,
                                                     map_location=torch.device(str_device)), strict=False)
                self.str_model.to(str_device)
                self.str_model.eval()
                print("Structure model weights loaded.")
            else:
                self.str_model = None



    def recognize(self, img, tokens=None, out_objects=False, out_cells=False,
                  out_html=False, out_csv=False):
        width, height = img.size
        out_formats = {}
        if self.str_model is None:
            print("No structure model loaded.")
            return out_formats
        print("out_html:", out_html)
        if not (out_objects or out_cells or out_html or out_csv):
            print("No output format specified")
            return out_formats

        # Transform the image how the model expects it
        img_tensor = structure_transform(img.convert("RGB"))

        # Run input_1 image through the model
        outputs = self.str_model([img_tensor.to(self.str_device)])

        # Post-process detected objects, assign class labels
        objects = outputs_to_objects(outputs, img.size, self.str_class_idx2name)
        # ------------------------------ 新增步骤 ------------------------------
        # 查找所有检测到的 "table row" 对象
        # table_rows = [obj for obj in objects if obj['label'] == 'table row']
        # segments = split_and_crop_by_overlap(img, table_rows,
        #                                      overlap_threshold=0.2,
        #                                      output_dir=self.output_img_dir)
        # all_cells = []
        # if table_rows:
        #     for seg_img, seg_rows, seg_offset in segments:
        table_rows = [obj for obj in objects if obj['label'] == 'table row']
        if width > height:
            segments = [(img, table_rows, 0)]
        else:
            segments = split_and_crop_by_overlap( img, table_rows, overlap_threshold=0.2,output_dir=self.output_img_dir)
        all_cells = []
        if table_rows:
            for seg_img, seg_rows, seg_offset in segments:
                # 1) 重新生成 tensor，跑模型
                img_t = structure_transform(seg_img.convert("RGB"))
                outputs = self.str_model([img_t.to(self.str_device)])
                objs = outputs_to_objects(outputs, seg_img.size, self.str_class_idx2name)

                # 2) 拿到每段的 table_structures，然后转 cells
                tables = objects_to_structures(objs, tokens, self.str_class_thresholds)
                for tbl in tables:
                    cells, _ = structure_to_cells(tbl, tokens)
                    # 3) 给每个 cell 记录 offset，后面生成 row_order 用
                    for cell in cells:
                        cell['row_offset'] = seg_offset
                    all_cells.append(cells)

            # 最终把所有段的 cells 合并扁平化
            # 这里假设你只处理一个表格，取 all_cells[0]；如果有多个表格，可依次输出
            out_formats['cells'] = all_cells[0]
        # -----------------------------------------------------------------------

        if out_objects:
            out_formats['objects'] = objects
        if not (out_cells or out_html or out_csv):
            return out_formats

        # Further process the detected objects so they correspond to a consistent table
        tables_structure = objects_to_structures(objects, tokens, self.str_class_thresholds)

        # Enumerate all table cells: grid cells and spanning cells
        tables_cells = [structure_to_cells(structure, tokens)[0] for structure in tables_structure]
        if out_cells:
            out_formats['cells'] = tables_cells

            # 生成唯一文件名
        timestamp = int(time.time() * 1000)
        process_id = f"proc_{timestamp}"
        output_dir = os.path.join(self.output_img_dir, process_id)
        os.makedirs(output_dir, exist_ok=True)

        # 保存原始图片
        original_path = os.path.join(output_dir, "original.jpg")
        img.save(original_path)

        # 执行裁剪时传递输出目录
        # img, adjusted_rows, crop_records = iterative_crop_by_overlap(
        #     img, table_rows,
        #     output_dir=output_dir,  # 传递输出目录
        #     overlap_threshold=0.2
        # )
        return tables_cells


import os
import numpy as np
from PIL import Image
from collections import defaultdict
from typing import List, Dict
from paddleocr import PaddleOCR

# def is_header_row(cells, threshold=0.8):
#     # 清理文本，去掉首尾空格
#     def clean_text(text):
#         return text.strip() if text else ""
#
#     # 预处理后的单元格文本列表
#     texts = [clean_text(cell["text"]) for cell in cells]
#
#     # 如果整行没有任何一个单元格包含数字，则认为不可能是表头
#     if not any(re.search(r'\d', text) for text in texts if text):
#         return False
#
#     # 针对非空文本，判断符合 is_header_text 特征的单元格比例
#     valid_texts = [text for text in texts if text]
#     if not valid_texts:
#         return False
#
#     header_count = sum(1 for text in valid_texts if is_header_text(text))
#     return header_count / len(valid_texts) >= threshold

import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    将全角字符（数字、字母、标点、空格）等 Unicode 变体
    规范成半角/常规 ASCII 形式，便于后续匹配。
    """
    # NFKC 会把全角数字、字母、空格、标点都转换成半角
    return unicodedata.normalize('NFKC', text)


def is_header_text(text: str,
                   filter_pure_punct: bool = False
                   ) -> bool:
    """
    判断单元格文本是否符合“表头”特征（改进版）。

    参数：
        text (str): OCR 识别出的原始单元格内容
        filter_pure_punct (bool): 是否排除“只有标点符号”的文本行

    步骤：
    1. 先做 Unicode NFKC 标准化，再 strip() 掉首尾空白；
    2. 如果 strip 后是真正的空 -> False；
    3. 如果全部由数字组成 -> False；
    4. （可选）如果只包含标点 -> False；
    5. 其他一律视为“表头” -> True。
    """
    # 1. 规范化 + 去首尾空白
    text = normalize_text(text).strip()
    if not text:
        return False

    # 2. 纯数字排除
    if text.isdigit():
        return False

    # 3. 可选：排除“只有标点”行
    if filter_pure_punct:
        # 去掉所有中英文、数字后，剩下的字符
        core = re.sub(r'[A-Za-z0-9\u4e00-\u9fff]', '', text)
        # 定义常见标点集合（可根据需要增删）
        puncts = set('.,:;·•‐‑–—()[]{}<>/\\|“”‘’"\'%&*+-=_…')
        # 如果剩余都在标点集合里，认为不是表头
        if core and all(ch in puncts for ch in core):
            return False

    # 否则，一律认为是表头
    return True

def extract_text(ocr_results):
    """提取OCR识别的文本（假设ocr_results是OCR返回的结果）"""
    extracted_text = []
    for image_data in ocr_results:
        # 检查 image_data 是否为 None，若是则跳过当前循环
        if image_data is None:
            continue
        for text_entry in image_data:
            if text_entry is not None:
                extracted_text.append(text_entry[1][0])  # 获取文字部分
            else:
                extracted_text.append("")
    return ' '.join(extracted_text)  # 组合成一个完整字符串


def ocr_table_to_structured_output(extracted_table, img_path, crop_records=[]) -> List[Dict]:
    """直接生成res2格式的结构化数据，并对所有页（包括第一页）全部识别"""
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True, gpu_id=0)
    img = Image.open(img_path)
    img_np = np.array(img)

    filename = os.path.basename(img_path)

    # 提取页码和子页标识（如 1a, 1b 等）
    # match = re.search(r"_(\d+)(?:_[a-zA-Z])?\.jpg$", filename)
    # if match:
    #     page_table_id = match.group(1)
    # else:
    #     page_table_id = "001"  # 默认值
    # filename = os.path.basename(img_path)
    filename = os.path.splitext(os.path.basename(img_path))[0]
    parts = filename.split('_')
    page_num = int(parts[-2])
    # 只提取数字页码，比如 “_6_a.jpg” 或 “_16b.jpg” 都拿数字部分
#     m = re.search(r"_(\d+)", filename)
#     page_num = int(m.group(1)) if m else 0
 # 三位宽零填充
    page_str = f"{page_num:03d}"

    res2_data = []
    row_dict = defaultdict(list)  # 按行号分组存储单元格

    for cell in extracted_table[0]:
        bbox = cell['bbox']
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(img_np.shape[1], int(bbox[2])), min(img_np.shape[0], int(bbox[3]))

        cell_img = img_np[y1:y2, x1:x2]
        ocr_result = ocr.ocr(cell_img, cls=True)
        # cell_text = extract_text(ocr_result) if ocr_result else ""

        raw_text = extract_text(ocr_result) if ocr_result else ""
         # 改为：只去掉换行回车，直接拼接
        cell_text = raw_text.replace('\r', '').replace('\n', '').strip()
        # cell_text = raw_text.replace('\r', ' ').replace('\n', ' ')
        row_nums = cell.get('row_nums', [0])
        col_nums = cell.get('column_nums', [0])

        for row_num in row_nums:
            row_dict[row_num].append({
                "col_num": min(col_nums),
                "text": cell_text,
                "is_header": cell.get('column header', False)
            })

    row_offset = sum([r['original_rows'] for r in crop_records])

    for row_num in sorted(row_dict.keys()):
        # adjusted_row_num = row_num + row_offset
        adjusted_row = row_num + row_offset

        sorted_cells = sorted(row_dict[row_num], key=lambda x: x["col_num"])

        # row_str = f"{adjusted_row + 1:03d}"
        # row_str = f"{adjusted_row_num + 1:03d}"
        header_row = all(
            not re.search(r'\d', cell["text"]) and is_header_text(cell["text"])
            for cell in sorted_cells
        )

        con = {}
        for col_idx, cell in enumerate(sorted_cells, 1):
            con[f"{row_num + 1}.{col_idx}"] = cell["text"].strip()



        res2_data.append({
            "row_order": f"{page_str}_{adjusted_row + 1:03d}",
            # "row_order": f"{page_table_id}_{adjusted_row_num + 1:03d}",
            "header_row": header_row,
            "con": con
        })

    return res2_data



def build_output_json(
    page_sum,
    outside_infos,   # 例如：[[{"box": [...], "txt": "..."}], [...]]
    table_rows ,
    error_page   # 例如：一个列表，列表里每个元素都包含 row_order, header_row, con 等
):
    # 这里假设需要填充的字段与示例一致
    output_data = {
        "code": "200",
        "message": "success",
        "data": {
            "agent_type": "unknow",
            "doc_type": "imagePDF",
            "error_page": error_page,
            "page_sum": page_sum,   # 从外部传入

            "res1": {
                "account_name": "",
                "account_num": "",
                "bank_name": "",
                "idcard_num": "",
                "outside_infos": outside_infos  # 从外部传入
            },

            "res2": table_rows  # 从外部传入
        }
    }

    return output_data


def save_json(output_data, save_path):
    # 将组装好的 Python 字典写出到 JSON 文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"JSON 已保存到: {save_path}")



# def process_head_first_image(head_dir: str) -> List[List[Dict]]:
#     """处理head文件夹中的第一个upper文件"""
#     ocr = PaddleOCR(use_angle_cls=True, lang="ch",use_gpu=True,gpu_id=0)
#
#     # 匹配文件名格式：前缀_页码_upper_序号.jpg
#     img_files = []
#     # for f in os.listdir(head_dir):
#     #     match = re.match(r"^.*_(\d+)+\.jpg$", f)
#     #     if match:
#     #         page_num = int(match.group(1))
#     #         img_files.append((f, page_num))
#      # 新版：所有 jpg 都进来，能提取数字就用页码，否则当做第 0 页
#
#     for f in os.listdir(head_dir):
#
#         if not f.lower().endswith('.jpg'):
#
#             continue
#         m = re.search(r"_(\d+)(?:[a-zA-Z])?\.jpg$", f)
#         page_num = int(m.group(1)) if m else 0
#         img_files.append((f, page_num))
#
#     if not img_files:
#         return []
#
#     # 按页码排序取第一个
#     img_files.sort(key=lambda x: x[1])
#     first_img_name, first_page = img_files[0]
#     first_img_path = os.path.join(head_dir, first_img_name)
#     print(f"[DEBUG] process_head_first_image: using head image '{first_img_name}' as page {first_page}")
#
#     # OCR处理
#     try:
#         # img = cv2.imread(first_img_path)
#         # result = ocr.ocr(img)
#         img = cv2.imread(first_img_path)
#
#         if img is None:
#         # cv2 读不到时，用 PIL
#
#             # from PIL import Image
#             #
#             # import numpy as np
#             pil = Image.open(first_img_path)
#             img = np.array(pil)
#             result = ocr.ocr(img)
#
#         if not result:
#            # OCR 竟然没返回任何行，直接空
#             print(f"[WARN] OCR returned empty for {first_img_name}")
#
#             return []
#     except Exception as e:
#         print(f"OCR处理失败：{str(e)}")
#         return []
#
#     # 组织表外信息
#     outside_infos = []
#     page_info = []
#     for line in result:
#         if not isinstance(line, list):
#             continue
#         for word_info in line:
#             if not isinstance(word_info, (list, tuple)) or len(word_info) < 2:
#                 continue
#             box = word_info[0]
#             text = word_info[1][0]
#             x_coords = [p[0] for p in box]
#             y_coords = [p[1] for p in box]
#             page_info.append({
#                 "box": [float(min(x_coords)), float(min(y_coords)),
#                         float(max(x_coords)), float(max(y_coords))],
#                 "txt": text.strip()
#             })
#
#     # 按垂直位置排序
#     page_info.sort(key=lambda x: x["box"][1])
#     outside_infos.append(page_info)
#     return outside_infos

def process_head_first_image(head_dir: str) -> list:
    """
    只识别 head_dir 中页码最小的那一张 .jpg 图片，返回 OCR 后的 outside_infos 列表。
    """
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True, gpu_id=0)

    def extract_page_num(fname: str) -> int:
        m = re.search(r'_(\d+)(?=[a-zA-Z]?\.jpg$)', fname)
        return int(m.group(1)) if m else float('inf')

    imgs = [f for f in os.listdir(head_dir) if f.lower().endswith('.jpg')]
    if not imgs:
        return []

    imgs.sort(key=extract_page_num)
    first_img = imgs[0]
    img_path = os.path.join(head_dir, first_img)

    # 先尝试用 cv2 读取，再 fallback 到 PIL
    img = cv2.imread(img_path)
    if img is None:
        img = np.array(Image.open(img_path))

    # 执行 OCR
    ocr_result = ocr.ocr(img, cls=True)
    if not ocr_result:
        # 当 OCR 返回 None 或空列表时，直接返回空
        return []

    outside_infos = []

    for line in ocr_result:
        # 跳过 None 或非列表项
        if not isinstance(line, (list, tuple)):
            continue
        for word_info in line:
            # 跳过 None 或结构不对的项
            if (not isinstance(word_info, (list, tuple))
                    or len(word_info) < 2
                    or word_info[0] is None
                    or word_info[1] is None):
                continue
            box, (txt, _) = word_info
            # box 可能是一个四点列表：[[x1,y1],...,[x4,y4]]
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            outside_infos.append({
                "box": [
                    float(min(x_coords)), float(min(y_coords)),
                    float(max(x_coords)), float(max(y_coords))
                ],
                "txt": txt.strip()
            })

    # 按纵坐标从上到下排序
    outside_infos.sort(key=lambda x: x["box"][1])
    return outside_infos

# import os, re
# from typing import Dict

def main_pipeline(input_pdf: str):
    # 1. 生成图片并分割（假设 diaoyong 现在返回 (image_dir, page_count, init_no_table)）
    image_dir, page_count, init_no_table = diaoyong(input_pdf)
    table_dir = os.path.join(image_dir, "table")
    head_dir  = os.path.join(image_dir, "head")
    if not os.path.exists(table_dir):
        os.makedirs(table_dir)
    if not os.path.exists(head_dir):
        os.makedirs(head_dir)
    # 2. 表外信息 OCR
    outside_infos = process_head_first_image(head_dir)
    print(f"[DEBUG] 外部信息 OCR 识别到：{outside_infos!r}")

    # 3. 表内信息识别
    all_res2 = []
    no_table_detected = []
    pipe = TableExtractionPipeline(
        output_img_dir=os.path.join(image_dir, "processed_tables"),
        str_device="cuda",
        str_config_path='../model/structure_config.json',
        str_model_path='../model/model_46.pth'
    )

    # 辅助函数：从文件名提取页码数字
    def extract_page_num(fname: str) -> int:
        # 支持 "_6.jpg", "_6a.jpg", "_16b.jpg" 这一类
        m = re.search(r'_(\d+)(?=[a-zA-Z]?\.jpg$)', fname)
        return int(m.group(1)) if m else -1

    # 先列出所有 .jpg，然后按页码排序
    table_files = sorted(
        [f for f in os.listdir(table_dir) if f.lower().endswith(".jpg")],
        key=extract_page_num
    )

    for table_file in table_files:
        img_path = os.path.join(table_dir, table_file)
        page_num = extract_page_num(table_file)

        img = Image.open(img_path)
        tables_cells = pipe.recognize(img, tokens=[], out_cells=True)
        crop_records = getattr(pipe, 'crop_records', [])

        # 这页如果完全没检测到任何 table_row，就记为无表格页
        if not tables_cells or len(tables_cells[0]) == 0:
            if page_num >= 1:
                no_table_detected.append(page_num)
            continue

        # 否则生成 res2 结构
        res2_part = ocr_table_to_structured_output(tables_cells, img_path, crop_records)
        all_res2.extend(res2_part)

    # 4. 合并、去重、排序所有无表格页
    # 4. 合并、去重、排序所有无表格页
    # —— 先把 init_no_table（可能是 NumPy array 或含 numpy.int32 的列表）里的元素都 cast 成 Python int
    init_list = [int(x) for x in init_no_table]
    error_pages = sorted(set(init_list + no_table_detected))

    all_res2.sort(key=lambda x: tuple(map(int, x['row_order'].split('_'))))

    # 5. 构建并保存最终 JSON
    page_count = int(page_count)
    final_json = build_output_json(
        page_sum=str(page_count).zfill(3),
        outside_infos=outside_infos,
        table_rows=all_res2,
        error_page=error_pages
    )

    pdf_basename   = os.path.splitext(os.path.basename(input_pdf))[0]
    output_path    = os.path.join(image_dir, f"{pdf_basename}.json")
    #save_json(final_json, output_path)
    shutil.rmtree(head_dir, ignore_errors=True)
    shutil.rmtree(table_dir, ignore_errors=True)
    shutil.rmtree(os.path.join(image_dir, "processed_tables"), ignore_errors=True)
    return final_json


# def main_pipeline(input_pdf: str) -> Dict:
#     # 生成图片并分割表格/表外信息
#     image_dir, page_count,error_page = diaoyong(input_pdf)
#     table_dir = os.path.join(image_dir, "table")
#     head_dir = os.path.join(image_dir, "head")
#
#     ############### 处理表外信息 ###############
#     outside_infos = process_head_first_image(head_dir)
#
#     ############### 处理表内信息 ###############
#     all_res2 = []
#     no_table_detected = []
#     pipe = TableExtractionPipeline(
#         output_img_dir=os.path.join(image_dir, "processed_tables"),  # 指定路径
#         str_device="cuda",
#         str_config_path='D:/code/table-transformer-main/src/structure_config.json',
#         str_model_path='D:\code\\table-transformer-main\models\model_46.pth'
#     )
#
#     # 按文件名顺序处理表格
#     table_files = sorted(
#         [f for f in os.listdir(table_dir) if f.endswith(".jpg")],
#         key=lambda x: (
#             # int(re.search(r"_(\d+)", x).group(1)),  # 提取页码
#             int(re.search(r"_(\d+)\.jpg$", x).group(1))  # 提取表格序号
#         )
#     )
#
#     for table_file in table_files:
#         img_path = os.path.join(table_dir, table_file)
#         print(f"Processing table: {table_file}")
#
#         # 表格结构识别
#         img = Image.open(img_path)
#         extracted_tables = pipe.recognize(img, tokens=[], out_cells=True)
#         if not tables_cells or len(tables_cells[0]) == 0:
#             if page_num is not None:
#                 no_table_detected.append(page_num)
#             continue
#         crop_records = pipe.crop_records if hasattr(pipe, 'crop_records') else []
#         # 直接获取res2结构
#         res2_part = ocr_table_to_structured_output(extracted_tables, img_path, crop_records)
#         all_res2.extend(res2_part)
#     page_count = int(page_count/2)
#     ############### 构建最终JSON ###############
#     final_json = build_output_json(
#
#         page_sum=str(page_count).zfill(3),
#         outside_infos=outside_infos,
#         table_rows=all_res2
#     )
#
#     # 保存结果
#     pdf_basename = os.path.splitext(os.path.basename(input_pdf))[0]  # 提取无扩展的文件名
#     output_filename = f"{pdf_basename}.json"  # 构造新的文件名
#     output_path = os.path.join(image_dir, output_filename)  # 保持原目录结构
#     save_json(final_json, output_path)
#     # return final_json
#     return output_path

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="将 PDF 文档拆分表格并输出结构化 JSON"
    # )
    # parser.add_argument(
    #     'input_pdf',
    #     help='输入 PDF 文件路径'
    # )
    # args = parser.parse_args()
    file_path ='D:\Google/UTrans-Net A Model for Short-Term Precipitation Prediction.pdf'
    result_path = main_pipeline(file_path)
    print(f"处理完成！结果保存至： {result_path}")

# if __name__ == "__main__":
#     input_path = r"C:\Users\10594\Desktop\cell测试问题\cell测试问题\62d62f35503eb.pdf"# 修改为您的PDF路径
#     result = main_pipeline(input_path)
#     print("处理完成！结果保存至：", os.path.join(os.path.dirname(input_path), "output/final_output.json"))
# if __name__ == "__main__":
#     import glob
#     import os
#
#     # 1. 设置你的输入目录（只要把所有 PDF 放到这里就行）
#     input_dir = r"C:\Users\10594\Desktop\cell测试问题\cell测试问题"
#     # 2. 遍历目录下所有 .pdf 文件
#     pdf_paths = glob.glob(os.path.join(input_dir, "*.pdf"))
#
#     if not pdf_paths:
#         print(f"在目录 {input_dir} 中没有找到任何 PDF 文件。")
#     else:
#         for pdf_path in pdf_paths:
#             try:
#                 # 调用主流程
#                 output_json = main_pipeline(pdf_path)
#                 print(f"[SUCCESS] 处理完成：{pdf_path} → {output_json}")
#             except Exception as e:
#                 print(f"[ERROR] 处理失败：{pdf_path}，原因：{e}")
