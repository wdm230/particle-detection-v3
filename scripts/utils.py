import xml.etree.ElementTree as ET

def parse_voc_xml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    boxes = []
    labels = []

    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(1)  # Assuming one class

    return boxes, labels
