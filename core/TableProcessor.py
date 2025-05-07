import os
import cv2
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import glob
import re

CLASS_LABEL_ANGLE_MAP = {
    0: 0, 1: 180, 2: 270, 3: 90
}


def get_image_paths(folder, exts=("*.jpg", "*.jpeg", "*.png")):
    def extract_page_number(path):
        stem = Path(path).stem
        match = re.search(r'_(\d+)', stem)
        return int(match.group(1)) if match else float('inf')

    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
        paths.extend(glob.glob(os.path.join(folder, ext.upper())))
    return sorted(paths, key=extract_page_number)


class TableProcessor:
    def __init__(self, input_path, seg_model_path, cls_model_path, alt_cls_model_path,
                 a, conf_threshold=0.7, cls_conf_threshold=0.5, dpi=300, max_size_mb=1.5,
                 timeout_seconds=30, debug=False):
        self.input_path = input_path
        self.seg_model_path = seg_model_path
        self.cls_model_path = cls_model_path
        self.alt_cls_model_path = alt_cls_model_path
        self.conf_threshold = conf_threshold
        self.cls_conf_threshold = cls_conf_threshold
        self.dpi = dpi
        self.max_size_mb = max_size_mb
        self.timeout_seconds = timeout_seconds
        self.debug = debug

        self.a = a
        self.b = set()  # 使用集合自动去重

        self.cls_model = None
        self.alt_cls_model = None
        self.seg_model = None

    def yolo_inference_and_extract(self, image_path):
        page_num = int(Path(image_path).stem.split('_')[-1])
        base_filename = Path(image_path).stem.rsplit('_', 1)[0]
        print(f"[进度] 处理图像：{image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"[警告] 图像读取失败：{image_path}")
            self.b.add(page_num)
            return False

        if self.seg_model is None:
            try:
                self.seg_model = YOLO(self.seg_model_path)
            except Exception as e:
                print(f"[错误] 加载分割模型失败：{e}")
                self.b.add(page_num)

                return False

        seg_results = self.seg_model(image_path, conf=self.conf_threshold, verbose=False)[0]
        no_table_detected = (
            not hasattr(seg_results, 'masks') or
            seg_results.masks is None or
            seg_results.masks.data is None or
            seg_results.masks.data.shape[0] == 0 or
            len(seg_results.boxes) == 0
        )

        if no_table_detected:
            print(f"[警告] 未检测到表格：{image_path}")
            self.b.add(page_num)

            return False

        mask_array = seg_results.masks.data.cpu().numpy()
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for i in range(mask_array.shape[0]):
            mask = cv2.resize(mask_array[i].astype(np.uint8), (image.shape[1], image.shape[0]))
            combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8) * 255

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"[警告] 无有效轮廓：{image_path}")
            self.b.add(page_num)
            return False

        table_dir = os.path.join(self.input_path, "table")
        head_dir = os.path.join(self.input_path, "head")
        os.makedirs(table_dir, exist_ok=True)
        os.makedirs(head_dir, exist_ok=True)

        idx = 0
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                continue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            x_min, y_min = np.maximum(np.min(box, axis=0), 0)
            x_max, y_max = np.minimum(np.max(box, axis=0), [image.shape[1], image.shape[0]])
            x_min_exp, x_max_exp = max(0, x_min -30), x_max + 35
            y_min_exp, y_max_exp = max(0, y_min - 10), y_max + 15
            roi = image[y_min_exp:y_max_exp, x_min_exp:x_max_exp].copy()
            if roi.size == 0:
                continue

            suffix_letter = chr(97 + idx)  # 'a', 'b', 'c',...
            new_filename = f"{base_filename}_{page_num}_{suffix_letter}.jpg"
            table_path = os.path.join(table_dir, new_filename)
            cv2.imwrite(table_path, roi)
            print(f"[保存] 表格区域：{table_path}")

            self.crop_and_save_upper_region(image, box, new_filename, head_dir)
            idx += 1

        if idx == 0:
            self.b.add(page_num)
            return False
        return True

    def crop_and_save_upper_region(self, image, box, filename, head_dir):
        sorted_pts = sorted(box, key=lambda p: p[1])
        top_pts = sorted(sorted_pts[:2], key=lambda p: p[0])
        y_line = int((top_pts[0][1] + top_pts[1][1]) / 2)
        cropped = image[:y_line, :]
        if np.mean(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)) > 253:
            print("[跳过] 上半图为空白")
            return
        out_path = os.path.join(head_dir, filename)
        cv2.imwrite(out_path, cropped)
        print(f"[保存] 上半区域：{out_path}")

    def post_process_output(self):
        folder_path = os.path.join(self.input_path, "table")
        print(f"[后处理] table 文件夹图像：{folder_path}")
        for image_path in get_image_paths(folder_path):
            print(f"[后处理] {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                continue
            _, angle = self.detect_orientation_with_conf(image)
            if angle == 90:
                corrected = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                corrected = cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                corrected = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                corrected = image
            cv2.imwrite(image_path, corrected)
        print("[后处理完成]")

    def detect_orientation_with_conf(self, image):
        if self.cls_model is None:
            self.cls_model = YOLO(self.cls_model_path)
        try:
            results = self.cls_model(image, conf=self.cls_conf_threshold, verbose=False, imgsz=224)[0]
            top1_label = results.probs.top1
            top1_conf = results.probs.top1conf
            if top1_conf < 0.6:
                if self.alt_cls_model is None:
                    self.alt_cls_model = YOLO(self.alt_cls_model_path)
                alt_results = self.alt_cls_model(image, conf=self.cls_conf_threshold, verbose=False, imgsz=224)[0]
                top1_label = alt_results.probs.top1
            angle = CLASS_LABEL_ANGLE_MAP.get(top1_label, 0)
            return top1_label, angle
        except Exception as e:
            print(f"[错误] 分类模型失败：{e}")
            return None, 0

    def clean_up_folder(self):
        print(f"[清理] 开始删除原始图片...")
        for filename in os.listdir(self.input_path):
            full_path = os.path.join(self.input_path, filename)
            if os.path.isfile(full_path) and filename.lower().endswith(".jpg"):
                if not full_path.startswith(os.path.join(self.input_path, "table")) and \
                   not full_path.startswith(os.path.join(self.input_path, "head")):
                    try:
                        os.remove(full_path)
                        print(f"[删除] {full_path}")
                    except Exception as e:
                        print(f"[错误] 无法删除 {full_path}：{e}")
        print(f"[清理完成] 原始图像已删除")


    def process(self):
        image_paths = get_image_paths(self.input_path)
        for img_path in image_paths:
            self.yolo_inference_and_extract(img_path)
        self.post_process_output()
        self.clean_up_folder()
        print(f"✅ 总页数 a = {self.a}")
        print(f"⛔ 无表格页码 b = {np.array(self.b)}")
        return self.input_path

    def get_summary(self):
        return self.a, np.array(sorted(self.b))  # 返回有序的唯一页码列表


if __name__ == '__main__':

    # === 步骤 1：设置路径 ===
    input_directory = r"C:\Users\77103\Desktop\test\1619593789496_1073352"  # 输入图片文件夹路径
    seg_model_path = r"D:\code\table-transformer-main\models\best.pt"  # 表格分割模型路径
    cls_model_path = r"C:\Users\77103\Desktop\lzpTableDetect\models\cls_step1.pt"  # 主分类模型路径
    alt_cls_model_path = r"C:\Users\77103\Desktop\lzpTableDetect\models\cls_step2.pt"  # 备用分类模型路径

    # === 步骤 2：设置总页数（即转换为图片的总张数） ===
    total_pages = 3  # 你自己统计转换出的图片总页数，比如 _1.jpg 到 _5.jpg

    # === 步骤 3：初始化 TableProcessor ===
    table_processor = TableProcessor(
        input_path=input_directory,
        seg_model_path=seg_model_path,
        cls_model_path=cls_model_path,
        alt_cls_model_path=alt_cls_model_path,
        a=total_pages,  # 设置页数 a
        conf_threshold=0.7,
        cls_conf_threshold=0.5,
        dpi=300,
        max_size_mb=1.5,
        timeout_seconds=30,
        debug=False
    )

    # === 步骤 4：开始处理 ===
    output_dir = table_processor.process()

    # === 步骤 5：获取结果（a 和 b）===
    a_val, b_array = table_processor.get_summary()
    print("✅ 处理完成")
    print(f"📄 总页数 a = {a_val}")
    print(f"⛔ 无表格页码 b = {b_array}")


