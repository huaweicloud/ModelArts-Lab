import numpy as np
from PIL import Image, ImageDraw, ImageFont
from xml.dom import minidom
import random
import cv2
import os

def generateXml(xml_path, boxes, w, h, d):
    impl = minidom.getDOMImplementation()

    doc = impl.createDocument(None, None, None)

    rootElement = doc.createElement('annotation')
    sizeElement = doc.createElement("size")
    width = doc.createElement("width")
    width.appendChild(doc.createTextNode(str(w)))
    sizeElement.appendChild(width)
    height = doc.createElement("height")
    height.appendChild(doc.createTextNode(str(h)))
    sizeElement.appendChild(height)
    depth = doc.createElement("depth")
    depth.appendChild(doc.createTextNode(str(d)))
    sizeElement.appendChild(depth)
    rootElement.appendChild(sizeElement)
    for item in boxes:
        objElement = doc.createElement('object')
        nameElement = doc.createElement("name")
        nameElement.appendChild(doc.createTextNode(str(item[0])))
        objElement.appendChild(nameElement)
        difficultElement = doc.createElement("difficult")
        difficultElement.appendChild(doc.createTextNode(str(0)))
        objElement.appendChild(difficultElement)
        bndElement = doc.createElement('bndbox')
        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(str(item[1])))
        bndElement.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(str(item[2])))
        bndElement.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(str(item[3])))
        bndElement.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(str(item[4])))
        bndElement.appendChild(ymax)
        objElement.appendChild(bndElement)
        rootElement.appendChild(objElement)

    doc.appendChild(rootElement)

    f = open(xml_path, 'w')
    doc.writexml(f, addindent='  ', newl='\n')
    f.close()

Index = 0

exp_path='./DeepLeague100K/origin_data/train'
def export(npz_file_name, exp_path):
    global Index
    np_obj = np.load(npz_file_name)
    print (len(np_obj['images']))
    for image, boxes in zip(np_obj['images'], np_obj['boxes']):
        img = Image.fromarray(image)
        img = np.array(img, dtype = np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        generateXml(exp_path + '/Annotations/' + str(Index) + '.xml', boxes, img.shape[0], img.shape[1], img.shape[2])
        cv2.imwrite(exp_path + '/Images/' + str(Index) + '.jpg', img)
        Index += 1


	
if __name__ == '__main__':
    root_path = './DeepLeague100K/clusters_cleaned/train/'
	npz_names = os.listdir(root_path)
	for item in npz_names:
		export(os.path.join(root_path, item), './DeepLeague100K/lol/train')
	root_path = './DeepLeague100K/clusters_cleaned/val/'
	npz_names = os.listdir(root_path)
	for item in npz_names:
		export(os.path.join(root_path, item), './DeepLeague100K/lol/eval')
