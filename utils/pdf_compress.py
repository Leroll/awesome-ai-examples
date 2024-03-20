import fitz
import os

def covert2pic(zoom):
	if os.path.exists('.pdf'):
		os.removedirs('.pdf')
	os.mkdir('.pdf')
	for pg in range(totaling):
		page  = doc.load_page(pg)
		zoom  = int(zoom)
		lurl  = '.pdf/%s.png' % str(pg+1)
		trans = fitz.Matrix(zoom/100.0,zoom/100.0)
		pm    = page.get_pixmap(matrix=trans,alpha=False)
		pm.save(lurl)
		print(page)
	doc.close()

def pic2pdf(obj):
	doc = fitz.open()
	for pg in range(totaling):
		img      = '.pdf/%s.png' % str(pg+1)
		imgdoc   = fitz.open(img)
		pdfbytes = imgdoc.convert_to_pdf()
		imgpdf   = fitz.open("pdf",pdfbytes)
		os.remove(img)
		doc.insert_pdf(imgpdf)
	if os.path.exists(obj):
		os.remove(obj)
	doc.save(obj)
	doc.close()

def pdfz(sor,obj,zoom):
	covert2pic(zoom)
	pic2pdf(obj)

if __name__ == "__main__":
    # ===============================================
    # 运行前修改以下三个参数, 测试了下，效果不是太好
	zoom = 20 #50代表缩小50%，200代表放大200%，100代表既不放大也不缩小
	sor  = 'xx.pdf' #输入文件名
	obj  = 'c_xx.pdf' #输出文件名 
    # ===============================================
	doc  = fitz.open(sor)
	totaling = doc.page_count
	pdfz(sor,obj,zoom)
	os.removedirs('.pdf')