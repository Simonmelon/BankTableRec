from core.PDFProcessor import PDFProcessor
from core.TableProcessor import TableProcessor

def diaoyong(input_pdf):
    # 此处传入pdf路径，调用pdf
    # input_pdf = r"D:\code\table-transformer-main\input_first\60eedc620057c.pdf"
    processor = PDFProcessor(input_pdf_path=input_pdf,zoom_num=8, dpi=200, max_size_mb=10, timeout_seconds=300)
    result_folder = processor.run()
    a = processor.a
    print(f"所有处理图片保存于：{result_folder}")

    # 下面将变量值提取和转移，方便table调用
    a = a
    path = result_folder

    # 下面是table的初始化，路径是上面传递的就ok了
    # === 步骤 1：设置路径 ===
    input_directory = path  # 输入图片文件夹路径
    seg_model_path = "../model/v11Lseg_416.pt"  # 表格分割模型路径
    cls_model_path = "../model/cls_step1.pt"  # 主分类模型路径
    alt_cls_model_path = "../model/cls_step2.pt"  # 备用分类模型路径

    # === 步骤 2：设置总页数（即转换为图片的总张数） ===
    total_pages = a  # 你自己统计转换出的图片总页数，比如 _1.jpg 到 _5.jpg

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
    a, b = table_processor.get_summary()
    print("✅ 处理完成")
    print(f"📄 总页数 a = {a}")
    print(f"⛔ 无表格页码 b = {b}")
    print("处理完之后的代码在",output_dir)
    return output_dir, a, b

# diaoyong(r"C:\Users\10594\Desktop\cell测试问题\cell测试问题\1215.pdf")