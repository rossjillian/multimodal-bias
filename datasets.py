import json
import argparse
import random
from tqdm import tqdm


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
            json_dict[id] = [k, caption_dict[id]]

    with open(write_name, 'w') as f:
        json.dump(json_dict, f)
    return json_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instances', type=str, help="File path to instances JSON")
    parser.add_argument('--annotations', type=str, help="File path to annotation JSON")
    parser.add_argument('--write_json', type=str, help="Name of JSON file to write extracted data")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    categories = ['bicycle', 'traffic light', 'sheep', 'tie', 'carrot', 'sink', 'book', 'remote', 'spoon', 'skis']
    labeled_dict = analyze_json(args.instances)
    # Returns dictionary of categories as keys and list of image ids as values
    category_dict = extract_categories(3000, categories, labeled_dict)
    # Returns dictionary of image ids and captions
    caption_dict = extract_annotations(args.annotations)
    # Saves JSON file of image id as key and list of [ category, [ captions ] ]
    save_image_text(category_dict, caption_dict, args.write_json)

