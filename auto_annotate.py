# auto_annotate.py - SIMPLIFIED
import cv2
import os
import glob
import xml.etree.ElementTree as ET
from xml.dom import minidom

def create_simple_annotation(filename, width, height, class_name):
    """Create simple XML annotation assuming hand is in center"""
    root = ET.Element("annotation")
    
    ET.SubElement(root, "filename").text = filename
    ET.SubElement(root, "folder").text = "gestures"
    
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    
    # Assume hand is in center 70% of image
    margin_w = int(width * 0.15)
    margin_h = int(height * 0.15)
    
    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, "name").text = class_name
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "difficult").text = "0"
    
    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(margin_w)
    ET.SubElement(bndbox, "ymin").text = str(margin_h)
    ET.SubElement(bndbox, "xmax").text = str(width - margin_w)
    ET.SubElement(bndbox, "ymax").text = str(height - margin_h)
    
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def main():
    print("🤖 STARTING AUTOMATIC ANNOTATION...")
    
    classes = ['up', 'down', 'left', 'right']
    annotation_count = 0
    
    for split in ['train', 'test']:
        for cls in classes:
            image_dir = f'dataset/augmented/{split}/{cls}'
            annot_dir = f'dataset/annotations/{split}/{cls}'
            os.makedirs(annot_dir, exist_ok=True)
            
            if not os.path.exists(image_dir):
                print(f"⚠️  Directory not found: {image_dir}")
                continue
                
            images = glob.glob(os.path.join(image_dir, '*.jpg'))
            print(f"📝 Annotating {len(images)} {cls} images for {split} set...")
            
            for img_path in images:
                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    
                    height, width = image.shape[:2]
                    filename = os.path.basename(img_path)
                    
                    # Create XML annotation
                    xml_content = create_simple_annotation(filename, width, height, cls)
                    
                    # Save annotation
                    xml_filename = filename.replace('.jpg', '.xml')
                    xml_path = os.path.join(annot_dir, xml_filename)
                    
                    with open(xml_path, 'w', encoding='utf-8') as f:
                        f.write(xml_content)
                    
                    annotation_count += 1
                    
                except Exception as e:
                    print(f"❌ Error annotating {img_path}: {e}")
    
    print(f"✅ ANNOTATION COMPLETE!")
    print(f"📄 Created {annotation_count} annotation files")
    print(f"📁 Saved to: dataset/annotations/")

if __name__ == "__main__":
    main()