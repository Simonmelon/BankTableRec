import os
import fitz
import datetime
from PIL import Image
import numpy as np


class PDFProcessor:
    def __init__(self, input_pdf_path, zoom_num=8, dpi=200, max_size_mb=10, timeout_seconds=300):
        """
        初始化PDF处理器类
        :param input_pdf_path: 输入的PDF文件路径（必须是单个PDF）
        :param zoom_num: 转换为图片时的缩放系数
        :param dpi: 设置转换为图片时的DPI
        :param max_size_mb: 转换后图片的最大大小（MB）
        :param timeout_seconds: PDF转换超时设置
        """
        if not os.path.isfile(input_pdf_path) or not input_pdf_path.lower().endswith('.pdf'):
            raise ValueError("输入路径必须是一个PDF文件！")

        self.input_pdf_path = input_pdf_path
        self.zoom_num = zoom_num
        self.dpi = dpi
        self.max_size_mb = max_size_mb
        self.timeout_seconds = timeout_seconds
        self.a = 0  # 用于记录PDF页数

    def pyMuPDF_fitz(self, pdfPath, imagePath, zoomNum, max_width=2000, max_height=2000):
        startTime_pdf2img = datetime.datetime.now()
        print("正在处理PDF文件，输出路径：", imagePath)

        pdfDoc = fitz.open(pdfPath)
        self.a = pdfDoc.page_count  # ✅ 记录页数
        estimated_time = self.a * 2
        print(f"[预估处理时间]: 共 {self.a} 页，约需 {estimated_time} 秒")

        for pg in range(self.a):
            page = pdfDoc[pg]
            mat = fitz.Matrix(zoomNum, zoomNum)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            width, height = pix.width, pix.height
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                mat = fitz.Matrix(zoomNum * scale, zoomNum * scale)
                pix = page.get_pixmap(matrix=mat, alpha=False)

            if not os.path.exists(imagePath):
                os.makedirs(imagePath)

            pg_str = f'{pg + 1:03}'
            pix.save(os.path.join(imagePath, f'{pg_str}.png'))

        print('pdf2img时间=', (datetime.datetime.now() - startTime_pdf2img).seconds, '秒')
        pass

    def pyMuBinaryzation(self, binaryzationpath):
        start = datetime.datetime.now()
        pic_name = [x for x in os.listdir(binaryzationpath) if x.lower().endswith(('jpg', 'png', 'jpeg'))]

        for i in pic_name:
            image = Image.open(os.path.join(binaryzationpath, i))
            picArray = np.array(image)

            red_data, green_data, blue_data = picArray[..., 0], picArray[..., 1], picArray[..., 2]
            red_green = red_data - green_data
            red_blue = red_data - blue_data

            black_150_index = np.where(
                (red_data < 160) & (green_data < 150) & (blue_data < 150) & (red_green != 0)
            )
            picArray[black_150_index] = [0, 0, 0]

            black_100_index = np.where(
                (red_data < 100) & (red_green == 0) & (red_blue == 0)
            )
            picArray[black_100_index] = [0, 0, 0]

            red_index = np.where((red_green > 10) & (red_blue > 10))
            picArray[red_index] = [255, 255, 255]

            white_200_index = np.where(
                (red_data > 200) & (green_data > 200) & (blue_data > 200)
            )
            picArray[white_200_index] = [255, 255, 255]

            im = Image.fromarray(picArray)
            im.save(os.path.join(binaryzationpath, i))
        print('pdfbinaryzation时间=', (datetime.datetime.now() - start).seconds, '秒')

    def compress_and_convert_to_jpg(self, image_path, output_path, quality=85):
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img.save(output_path, 'JPEG', quality=quality)

    def process_pdf_file(self, pdf_file, output_dir):
        print(f"开始处理文件: {pdf_file}")
        filename, _ = os.path.splitext(os.path.basename(pdf_file))
        output_pdf_folder = os.path.join(output_dir, filename)
        if not os.path.exists(output_pdf_folder):
            os.makedirs(output_pdf_folder)

        self.pyMuPDF_fitz(pdf_file, output_pdf_folder, self.zoom_num)
        self.pyMuBinaryzation(output_pdf_folder)

        for i, image_file in enumerate(os.listdir(output_pdf_folder)):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(output_pdf_folder, image_file)
                new_image_path = os.path.join(output_pdf_folder, f"{filename}_{i + 1}.jpg")
                self.compress_and_convert_to_jpg(image_path, new_image_path)
                os.remove(image_path)

        print(f"{filename} 处理完成！")
        return output_pdf_folder

    def run(self):
        """
        运行PDF处理，返回生成的处理结果文件夹路径
        """
        output_dir = os.path.dirname(self.input_pdf_path)
        return self.process_pdf_file(self.input_pdf_path, output_dir)


if __name__ == '__main__':
    input_pdf = r"C:\Users\77103\Desktop\test\1619593789496_1073352.pdf"
    processor = PDFProcessor(input_pdf_path=input_pdf, zoom_num=8, dpi=200, max_size_mb=10, timeout_seconds=300)
    result_folder = processor.run()
    a = processor.a
    print(f"所有处理图片保存于：{result_folder}")
    print(f"页数：{a}")
