from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont # 字体类


# 注册中文字体
# pdfmetrics.registerFont(UnicodeCIDFont("ukai"))
pdfmetrics.registerFont(TTFont('SimSun', 'SimSun.ttf'))

def txt_to_pdf(input_file, output_file):
    # 创建一个新的PDF文档
    c = canvas.Canvas(output_file)
    # 设置字体和字号
    c.setFont('SimSun', 10)
    
    # 打开TXT文件并逐行读取
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    c.drawString(10, 750, lines)
    c.showPage()

    # 将TXT文件每行写入PDF中
    # for line in lines:
    #     print(line)
    #     c.drawString(10, 750, line)
    #     c.showPage()

    # 保存并关闭PDF文档
    c.save()

# 输入和输出文件的路径
input_file = 'results.txt'
output_file = 'output.pdf'

# 调用函数进行转换
txt_to_pdf(input_file, output_file)
