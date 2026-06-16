# datasets/convert_labels.py
#
# WHAT THIS DOES:
#   Converts all VOC XML label files to YOLO .txt format.
#
# WHY IT IS NEEDED:
#   The FBD-SV-2024 dataset stores bounding boxes in VOC XML format:
#     <xmin> <ymin> <xmax> <ymax>  — all in absolute PIXELS
#
#   YOLOv8 needs labels in YOLO format:
#     class_id  center_x  center_y  width  height  — all as FRACTIONS (0.0 to 1.0)
#
#   This script reads every .xml in data/labels/train/ and data/labels/val/
#   and writes a matching .txt file into data/images/train/ and data/images/val/
#   (same folder as the images, same base filename — that's how YOLO finds labels).
#
# HOW TO RUN:
#   python datasets/convert_labels.py --data_root data/
#
# EXPECTED RESULT:
#   data/images/train/bird_1_000000.txt   ← created for each .jpg
#   data/images/val/bird_1_000000.txt
#   etc.

import os
import xml.etree.ElementTree as ET   # reads XML files
import argparse


def convert_one_xml(xml_path, out_txt_path):
    """
    Reads one VOC XML label file and writes one YOLO .txt label file.

    xml_path    : path to the input .xml file
    out_txt_path: path to write the output .txt file

    Returns the number of bird boxes written.
    """

    # Parse the XML into a tree we can query
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Read image size from the XML (needed to turn pixels into fractions)
    size_node = root.find("size")
    img_w = int(size_node.find("width").text)
    img_h = int(size_node.find("height").text)

    lines = []  # will hold one YOLO line per bird box

    # Loop over every <object> tag in the XML — each is one bird box
    for obj in root.findall("object"):

        # Class id is always 0 because there is only one class (bird)
        class_id = 0

        # Read the pixel bounding box corners
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert to YOLO format — divide by image size to get fractions
        cx = (xmin + xmax) / 2.0 / img_w   # center x
        cy = (ymin + ymax) / 2.0 / img_h   # center y
        bw = (xmax - xmin)       / img_w   # box width
        bh = (ymax - ymin)       / img_h   # box height

        # Build the YOLO line: "0 cx cy bw bh"
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    # Write all lines to the output .txt file
    # If there are no birds in this frame, we write an empty file (that's correct for YOLO)
    with open(out_txt_path, "w") as f:
        f.write("\n".join(lines))

    return len(lines)


def convert_split(xml_dir, img_dir):
    """
    Converts all .xml files in xml_dir and writes .txt files into img_dir.

    xml_dir : folder containing the XML label files  (e.g. data/labels/train/)
    img_dir : folder containing the image .jpg files  (e.g. data/images/train/)
              — the .txt files are written here, alongside the images
    """

    # Make sure the image folder exists (it must exist after frame extraction)
    if not os.path.isdir(img_dir):
        print(f"  WARNING: image folder does not exist yet: {img_dir}")
        print(f"  Run frame extraction first, then re-run this script.")
        return 0, 0

    total_files = 0
    total_boxes = 0

    # Loop over every file in the XML folder
    for filename in sorted(os.listdir(xml_dir)):

        # Skip anything that is not an XML file
        if not filename.endswith(".xml"):
            continue

        xml_path = os.path.join(xml_dir, filename)

        # Output .txt path: same folder as the images, same base name, .txt extension
        base_name   = os.path.splitext(filename)[0]   # "bird_1_000000"
        txt_filename = base_name + ".txt"              # "bird_1_000000.txt"
        out_txt_path = os.path.join(img_dir, txt_filename)

        n_boxes = convert_one_xml(xml_path, out_txt_path)
        total_files += 1
        total_boxes += n_boxes

    return total_files, total_boxes


def run(data_root):
    """
    Converts labels for both the train and val splits.
    data_root : root data folder (e.g. "data/")
    """

    print("=" * 50)
    print("  Converting VOC XML labels → YOLO .txt format")
    print("=" * 50)

    # ── Train split ───────────────────────────────────────────
    xml_train = os.path.join(data_root, "labels", "train")
    img_train = os.path.join(data_root, "images", "train")
    print(f"\n[Train]")
    print(f"  XML source : {xml_train}")
    print(f"  TXT output : {img_train}")
    n_files, n_boxes = convert_split(xml_train, img_train)
    print(f"  Converted  : {n_files} label files, {n_boxes} bird boxes")

    # ── Val split ─────────────────────────────────────────────
    xml_val = os.path.join(data_root, "labels", "val")
    img_val = os.path.join(data_root, "images", "val")
    print(f"\n[Val]")
    print(f"  XML source : {xml_val}")
    print(f"  TXT output : {img_val}")
    n_files, n_boxes = convert_split(xml_val, img_val)
    print(f"  Converted  : {n_files} label files, {n_boxes} bird boxes")

    print("\n" + "=" * 50)
    print("  Done. Run verify_label.py next to check one box visually.")
    print("=" * 50)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FBD-SV-2024 XML labels to YOLO format")
    parser.add_argument("--data_root", default="data/", help="Root data folder (default: data/)")
    args = parser.parse_args()

    run(args.data_root)
