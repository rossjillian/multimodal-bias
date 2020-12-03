import json
import argparse
import random
from tqdm import tqdm
import os
import shutil
import re
from PIL import Image


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
    :return: returns directory with keys as the category name and value of a list of image IDs
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


def parse_image_dir(image_dir, image_json, new_dir):
    """
    Copies images from original COCO directory image_dir to new_dir with COCO-10S images
    :return: 1 on success
    """
    with open(image_json) as f:
        image_dict = json.load(f)

    for file in tqdm(os.listdir(image_dir)):
        match = re.search('COCO_train2014_(\d+).jpg', file)
        image_num = match.group(1)
        if image_num in image_dict.keys():
            shutil.copy(os.path.join(image_dir, file), os.path.join(new_dir, file))
    return 1


def process_coco_s(json_file):
    """
    Processes COCO-10S JSON - extra helper for after-the-fact modifications
    :param json_file: file path to COCO-10S JSON
    :return: category dictionary
    """
    category_dict = {}
    with open(json_file) as f:
        image_id_dict = json.load(f)

    for k, v in image_id_dict.items():
        if v[0] not in category_dict.keys():
            category_dict[v[0]] = [k]
        else:
            category_dict[v[0]].append(k)
    return category_dict


def color_split(categories, category_dict):
    """
    Randomly choose 1/2 of categories to be 95% BW and 5% C; the other half will be 5% BW and 95% C
    """
    split_dict = {}
    # Leaving same seed for reproducability
    seed = 4
    random.Random(seed).shuffle(categories)
    # Assign grayscale and color split
    i = 0
    for k, v in tqdm(category_dict.items()):
        # We expect total to be 3000 for each category
        total = len(v)
        # 95% and 5% split
        split = int(total * 0.05)
        # Shuffle image IDs in category
        random.Random(seed).shuffle(v)
        # 95% BW, 5% C
        if i % 2 != 0:
            split = total - split

        for j in range(split):
            name = 'data/coco-10s-train/COCO_train2014_' + v[j] + '.jpg'
            original = Image.open(name)
            grey = original.convert('L')
            # Save to same name
            grey.save(name)
            split_dict[name] = 'grey'
        i += 1

    with open('data/coco-10s-grey.json', 'w') as f:
        json.dump(split_dict, f)

    return split_dict


def create_coco_s(args, categories):
    """
    Creates data/coco-10s-train directory
    """
    if not os.path.isdir(args.new_image_dir):
        os.mkdir(args.new_image_dir)

    # Returns directory with keys as the category name and value of a list of image IDs
    labeled_dict = analyze_json(args.instances)
    # Picks 3000 random image IDs in each category
    category_dict = extract_categories(3000, categories, labeled_dict)
    # Returns dictionary of image ids and captions
    caption_dict = extract_annotations(args.captions)
    # Saves JSON file of image id as key and list of [ category, [ captions ] ]
    save_image_text(category_dict, caption_dict, args.write_json)
    # Uses saved JSON to copy correct COCO files
    parse_image_dir(args.image_dir, os.path.join('data', args.write_json), 'data/coco-10s-train')
    # Do BW/C split
    color_split(categories, category_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    # Original COCO data args
    parser.add_argument('--instances', type=str, default='data/annotations/instances_train2014.json',
                        help="File path to original COCO instances JSON")
    parser.add_argument('--captions', type=str, default='data/annotations/captions_train2014.json',
                        help="File path to original COCO instances annotation JSON")
    parser.add_argument('--image_dir', type=str, default='data/train2014', help="Directory of COCO images")
    # COCO-10S data args
    parser.add_argument('--write_json', type=str, default='data/coco-10s.json',
                        help="Name of JSON file to write extracted COCO-10S data")
    parser.add_argument('--new_image_dir', type=str, default='data/coco-10s-train', help='Directory for COCO-10S data')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    categories = ['bicycle', 'traffic light', 'sheep', 'tie', 'carrot', 'sink', 'book', 'remote', 'spoon', 'skis']
    create_coco_s(args, categories)
