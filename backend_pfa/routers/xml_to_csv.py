import xml.etree.ElementTree as ET
import pandas as pd

tree = ET.parse("video3_annotee.xml")
root = tree.getroot()

rows = []

for track in root.findall("track"):

    if track.attrib.get("label") != "plate":
        continue

    plate_id = track.attrib.get("id")

    for box in track.findall("box"):

        frame = int(box.attrib["frame"])

        # ✅ FIX IMPORTANT : lire les attributes correctement
        immat = ""

        for attr in box.findall("attribute"):
            if attr.attrib.get("name") == "immatriculation":
                if attr.text is not None:
                    immat = attr.text.strip()

        rows.append([
            frame,
            int(plate_id),
            immat,
            float(box.attrib["xtl"]),
            float(box.attrib["ytl"]),
            float(box.attrib["xbr"]),
            float(box.attrib["ybr"]),
        ])

df = pd.DataFrame(
    rows,
    columns=["frame", "plate_id", "immatriculation", "x1", "y1", "x2", "y2"]
)

df.to_csv("video3_annotatee.csv", index=False, encoding="utf-8-sig")