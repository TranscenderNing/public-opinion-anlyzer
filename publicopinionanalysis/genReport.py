from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont # 字体类


# 注册中文字体
# pdfmetrics.registerFont(UnicodeCIDFont("ukai"))
pdfmetrics.registerFont(TTFont('SimSun', 'SimSun.ttf'))

# 创建一个新的PDF文件
pdf_file = "report.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)

# 设置页面标题（中文）
c.setFont("SimSun", 16)
c.drawString(50, 750, "报告标题")

# 添加内容（中文）
c.setFont("SimSun", 12)
# 打开TXT文件并逐行读取
with open('results.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()


h = 800
for line in lines:
    print(line,type(line))
    c.drawString(50, h, line)
    h += 20

# 保存并关闭PDF文件
c.save()

