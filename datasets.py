import json
import argparse
import random
from tqdm import tqdm
import os
import shutil
import re


def extract_annotations(annotation_file):
    """
    :param annotation_file: JSON
    :return: dictionary of image IDs as keys and the caption as the value
    """
    caption_dict = {}
    with open(annotation_file) as f:
        annotation_dict = json.load(f)
        for annotation in annotation_dict['annotations']:
            if annotation['image_id'] not in caption_dict.keys():
                caption_dict[annotation['image_id']] = [annotation['caption']]
            else:
                caption_dict[annotation['image_id']].append(annotation['caption'])
    return caption_dict


def extract_categories(num_per_class, categories, mapped_categories):
    """
    :param num_per_class: number of images per category
    :param categories: list of categories
    :return: dictionary of category ids with num_per_class randomly selected images per category
    """
    filtered_dict = {key: mapped_categories[key] for key in categories}
    for k, v in filtered_dict.items():
        filtered_dict[k] = random.sample(v, num_per_class)
    return filtered_dict


def analyze_json(json_file):
    """
    :param json_file: JSON of annotations and categories
    :return: returns count of each category in the JSON
    """
    labeled_category_dict = {}
    category_dict = {}
    # Map image IDs to category
    with open(json_file) as f:
        annotation_dict = json.load(f)
        for i in tqdm(annotation_dict['annotations']):
            if i['category_id'] not in category_dict.keys():
                category_dict[i['category_id']] = [i['image_id']]
            else:
                category_dict[i['category_id']].append(i['image_id'])

    # Map keys (id numbers) to category name
    for i in annotation_dict['categories']:
        labeled_category_dict[i['name']] = category_dict[i['id']]
    return labeled_category_dict


def save_image_text(category_dict, caption_dict, write_name):
    """
    :param category_dict:
    :param caption_dict:
    :return:
    """
    json_dict = {}
    for k, v in category_dict.items():
        for id in v:
            json_dict[str(id).zfill(12)] = [k, caption_dict[id]]

    with open(write_name, 'w') as f:
        json.dump(json_dict, f)
    return json_dict


def create_coco_s(args, categories):
    labeled_dict = analyze_json(args.instances)
    # Returns dictionary of categories as keys and list of image ids as values
    category_dict = extract_categories(3000, categories, labeled_dict)
    # Returns dictionary of image ids and captions
    caption_dict = extract_annotations(args.captions)
    # Saves JSON file of image id as key and list of [ category, [ captions ] ]
    save_image_text(category_dict, caption_dict, args.write_json)


def parse_image_dir(image_dir, image_json, new_dir):
    with open(image_json) as f:
        image_dict = json.load(f)

    for file in tqdm(os.listdir(image_dir)):
        match = re.search('COCO_train2014_(\d+).jpg', file)
        image_num = match.group(1)
        if image_num in image_dict.keys():
            shutil.move(os.path.join(image_dir, file), new_dir, file)

    return 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instances', type=str, default='data/annotations/instances_train2014.json', help="File path to instances JSON")
    parser.add_argument('--captions', type=str, default='data/annotations/captions_train2014.json', help="File path to annotation JSON")
    parser.add_argument('--write_json', type=str, default='coco-10s.json', help="Name of JSON file to write extracted data")
    parser.add_argument('--image_dir', type=str, default='data/train2014', help="Directory of images")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    categories = ['bicycle', 'traffic light', 'sheep', 'tie', 'carrot', 'sink', 'book', 'remote', 'spoon', 'skis']
    parse_image_dir(args.image_dir, os.path.join('data', args.write_json), 'data/coco-10s-train')

