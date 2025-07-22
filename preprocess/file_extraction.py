import os
from typing import List, Dict
import re
from pythainlp.util import normalize
from datetime import datetime
import json
from datasets import Dataset, Features, Sequence, Value, Image as HFImage
from PIL import Image, UnidentifiedImageError
import ast

from preprocess.conversation import convert_to_conversation, convert_to_conversation_test, INSTRUCTION

TARGET_SIZE = (256, 256)

# Convert Gregorian date to Thai date
def thai_parse_date(input_text):
    gregorian_date = input_text
    gregorian_date_obj = datetime.strptime(gregorian_date, "%Y%m%d")

    # Convert to Thai date format
    thai_year = gregorian_date_obj.year + 543
    thai_date = gregorian_date_obj.strftime(f"%d %B {thai_year}")

    thai_date = thai_date.replace("January", "มกราคม")
    thai_date = thai_date.replace("February", "กุมภาพันธ์")
    thai_date = thai_date.replace("March", "มีนาคม")
    thai_date = thai_date.replace("April", "เมษายน")
    thai_date = thai_date.replace("May", "พฤษภาคม")
    thai_date = thai_date.replace("June", "มิถุนายน")
    thai_date = thai_date.replace("July", "กรกฎาคม")
    thai_date = thai_date.replace("August", "สิงหาคม")
    thai_date = thai_date.replace("September", "กันยายน")
    thai_date = thai_date.replace("October", "ตุลาคม")
    thai_date = thai_date.replace("November", "พฤศจิกายน")
    thai_date = thai_date.replace("December", "ธันวาคม")
    return thai_date

def extract_date_from_filename(filename):
    """
    Extracts the date from a filename in the format YYYYMMDD.

    Args:
      filename: The name of the file.

    Returns:
      The extracted date as a string in YYYYMMDD format, or None if no date is found.
    """
    match = re.search(r"\d{8}", filename)  # Find an 8-digit number (YYYYMMDD)
    if match:
      date_text = match.group(0)
      date_text = thai_parse_date(date_text)
      return date_text
    return None

def clean_text(text: str) -> str:

    # Pattern to detect URLs
    url_pattern = re.compile(r'https?://\S+')

    # Pattern to detect lines that start with a number and end with "น."
    date_line_pattern = re.compile(r'^\s*\d+.*น\.\s*$')

    # Process each line
    first_cleaned_lines = []
    for line in text.splitlines():
        stripped_line = line.strip()
        # Remove lines that are solely a number
        if stripped_line.isdigit():
            continue
        # Remove lines containing any URL
        if url_pattern.search(line):
            continue
        # Remove lines that match the date/time pattern
        if date_line_pattern.match(line):
            continue

        line = line.replace(" า"," ำ")
        line = line.replace("ต ่า", "ต่ำ")
        line = line.replace("ก ำ", "กำ")
        line = line.replace("ก  ำ", "กำ")
        line = line.replace("ดำห์", "ดาห์")
        line = line.replace("มำ", "มา")

        first_cleaned_lines.append(line)

    first_cleaned_text = "\n".join(first_cleaned_lines)

    cleaned_lines = []
    for line in first_cleaned_text.splitlines():
        if "ที่มา:" in line:
            continue
        if "สภาพอากาศ" == line.strip():
            continue
        if "สัปดาห์ที่ผ่านมา" == line.strip():
            continue
        if "ข้อมูลเพิ่มเติม:" == line.strip():
            continue
        if "สัปดาห์ที่ผ่านมาสภาพอากาศ" == line.strip():
            continue
        if "ลักษณะกลุ่มเมฆจากภาพถ่ายดาวเทียม" in line.strip():
            continue
        if "กลุ่มเมฆและแผนที่อากาศ" == line.strip():
            continue
        if "ภาพแผนที่อากาศ กรมอุตุนิยมวิทยา" == line.strip():
            continue
        if "Digital Typhoon" in line.strip():
            continue

        line.strip()

        cleaned_lines.append(line)

    output_text = " ".join(cleaned_lines)

    output_text = normalize(output_text)

    return output_text.strip()

def create_dataset_list(mapped_images_folder: str) -> List[Dict[str, List]]:
    """Reads text and image files from mapped_images and returns a list of dictionaries for Dataset.from_list()."""
    dataset = []

    text_folder = os.path.join(mapped_images_folder, "text")
    img_folder = os.path.join(mapped_images_folder, "img_final")
    air_img_folder = os.path.join(mapped_images_folder, "img_air_pressure")
    img_metadata_folder = os.path.join(mapped_images_folder, "img_metadata")

    if not os.path.exists(text_folder) or not os.path.exists(img_folder) or not os.path.exists(img_metadata_folder):
      print(f"Skipping {mapped_images_folder} (Missing text/ or img/ folder)")
      return dataset  # Return empty dataset if folders don't exist

    # Read all text files
    for text_file in os.listdir(text_folder):
      if text_file.endswith(".txt"):
        base_name = os.path.splitext(text_file)[0]
        text_file_path = os.path.join(text_folder, text_file)

        # Read text content
        with open(text_file_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        # Find matching images
        matched_images = [
            os.path.join(img_folder, img_file)
            for img_file in os.listdir(img_folder)
            if img_file.startswith(base_name) and img_file.lower().endswith((".jpg", ".png", ".jpeg"))
        ] + [
            os.path.join(air_img_folder, img_file)
            for img_file in os.listdir(air_img_folder)
            if img_file.startswith(base_name) and img_file.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        matched_images = sorted(matched_images)

        matched_images_metadata = []
        # Find matching images
        for img_metadata_file in sorted(os.listdir(img_metadata_folder)):
          if img_metadata_file.startswith(base_name) and img_metadata_file.lower().endswith((".json")):
            # Read json file
            with open(os.path.join(img_metadata_folder, img_metadata_file), 'r') as f:
              data = json.load(f)

              # Extract counts per image
              for image, details in data.items():
                cloud_count = sum(1 for det in details['detections'] if det['class_id'] == 0)
                typhoon_count = sum(1 for det in details['detections'] if det['class_id'] == 1)
                matched_images_metadata.append({"Cloudy": cloud_count, "Typhoon": typhoon_count})

        report_date = extract_date_from_filename(base_name)
        caption = clean_text(caption)

        if matched_images:
          dataset.append({"image": matched_images, "text": caption, "filename": base_name, "reportdate": report_date, 'image_metadata': matched_images_metadata})

    print(f"✅ Total records created: {len(dataset)}")
    return dataset

def split_train_val_test(dataset, train_json_path="train.json", validation_json_path="validation.json", test_json_path="test.json"):
    """Splits the dataset into train, validation, and test sets."""
    train_dataset = []
    validation_dataset = []
    test_dataset = []

    for item in dataset:
        filename = item['filename']
        # print(filename)
        if '202504' in filename  or '202505' in filename:
            test_dataset.append(item)
        elif '202502' in filename or '202503' in filename:
            validation_dataset.append(item)
        else:
            train_dataset.append(item)

    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_dataset, f, ensure_ascii=False, indent=4)

    with open(validation_json_path, "w", encoding="utf-8") as f:
        json.dump(validation_dataset, f, ensure_ascii=False, indent=4)

    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_dataset, f, ensure_ascii=False, indent=4)

    print(f"JSON file saved at: {train_json_path}, {test_json_path}, {validation_json_path}")
    
    def open_json_dataset(json_path):
        # Load JSON data
        with open(json_path, "r") as f:
            dataset_list = json.load(f)
        return dataset_list

    train_dataset_list = open_json_dataset(train_json_path)
    validation_dataset_list = open_json_dataset(validation_json_path)
    test_dataset_list = open_json_dataset(test_json_path)

    # Define dataset schema
    features = Features({
        "image": Sequence(HFImage()),
        "text": Value("string"),
        "filename": Value("string"),
        "reportdate": Value("string"),
        "image_metadata": Value("string")
    })

    # Resize and load images
    def load_image(example):
        resized_images = []
        for img in example["image"]:
            try:
                img = img.convert("RGB")
                img = img.resize(TARGET_SIZE)
                resized_images.append(img)
            except UnidentifiedImageError:
                print(f"Error loading image: {img}")
                # Optionally, you can append a placeholder or None, or skip the image
                # For now, we'll just skip the problematic image
                pass
        example["image"] = resized_images
        return example

    # Refine image metadata
    def refine_image_metadata(sample):
        metadata_list = ast.literal_eval(sample["image_metadata"])
        # Format each entry with "วันที่ {i}"
        lines = [
            f"วันที่ {i+1}: {entry}"
            for i, entry in enumerate(metadata_list)
        ]

        # Join lines into a single string
        sample["image_metadata"] = "\n".join(lines)
        return sample
    
    train_dataset = Dataset.from_list(train_dataset_list, features=features).map(load_image)
    validation_dataset = Dataset.from_list(validation_dataset_list, features=features).map(load_image)
    test_dataset = Dataset.from_list(test_dataset_list, features=features).map(load_image)

    train_dataset = train_dataset.map(refine_image_metadata)
    validation_dataset = validation_dataset.map(refine_image_metadata)
    test_dataset = test_dataset.map(refine_image_metadata)
    
    train_conversation_dataset = [convert_to_conversation(sample, INSTRUCTION) for sample in train_dataset]
    validation_conversation_dataset = [convert_to_conversation(sample, INSTRUCTION) for sample in validation_dataset]
    test_conversation_dataset = [convert_to_conversation_test(sample, INSTRUCTION) for sample in test_dataset]
    
    return train_conversation_dataset, validation_conversation_dataset, test_conversation_dataset