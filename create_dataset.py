import json
import argparse
import random
from tqdm import tqdm
import os
import shutil
import re
from PIL import Image
from pycocotools.coco import COCO
import copy


def map_category_name(json_file):
    """
    :param json_file: file path to COCO instances JSON file
    :return: directory with keys as the category name and value of a list of image IDs
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
                if not i['image_id'] in category_dict[i['category_id']]:
                    category_dict[i['category_id']].append(i['image_id'])

    category_labels = {}
    # Map keys (id numbers) to category name
    for i in annotation_dict['categories']:
        category_labels[i['name']] = i['id']
        labeled_category_dict[i['name']] = category_dict[i['id']]

    return labeled_category_dict, category_labels


def extract_captions(caption_file, instance_file):
    """
    :param caption_file: file path for caption file
    :return: { image_id: [caption, [boxes]] }
    """
    coco = COCO(instance_file)
    caption_dict = {}
    with open(caption_file) as f:
        annotation_dict = json.load(f)
        for annotation in annotation_dict['annotations']:
            img_id = annotation['image_id']
            # Get category bounding box
            ann_id = coco.getAnnIds(imgIds=img_id)
            coco_ann = coco.loadAnns(ann_id)
            boxes = []
            for i in range(len(coco_ann)):
                xmin = coco_ann[i]['bbox'][0]
                ymin = coco_ann[i]['bbox'][1]
                xmax = xmin + coco_ann[i]['bbox'][2]
                ymax = ymin + coco_ann[i]['bbox'][3]
                boxes.append([coco_ann[i]['category_id'], [xmin, ymin, xmax, ymax]])
            if img_id not in caption_dict.keys():
                caption_dict[annotation['image_id']] = [[annotation['caption']], boxes]
            else:
                # Make sure caption and bbox are not repeated
                if annotation['caption'] not in caption_dict[annotation['image_id']][0]:
                    caption_dict[annotation['image_id']][0].append(annotation['caption'])
                for box in boxes:
                    if box not in caption_dict[annotation['image_id']][1]:
                        caption_dict[annotation['image_id']][1].append(box)

    return caption_dict


def combine_captions_category(category_dict, caption_dict, category_labels):
    """
    :return: dictionary with keys as category name and value of a dict of image IDs and "valid" captions
    { category_name: {image_id: [captions, bbox], image_id: [captions, bbox], ...}, ... }
    """
    filtered_dict = {}
    # Iterate over categories, value is list of image IDs
    for k, v in category_dict.items():
        inner_dict = {}
        # Iterate over image IDs in each category
        for id in v:
            caption_list = []
            # Get list of captions for image
            captions = caption_dict[id][0]
            for caption in captions:
                test_caption = caption.lower()
                # For each caption, check if contains key (category name)
                if k in test_caption:
                    caption_list.append(test_caption)

            box_list = []
            # Get list of bbox for image
            bbox = caption_dict[id][1]
            for box in bbox:
                # Only select bounding box of category
                if box[0] == category_labels[k]:
                    box_list.append(box[1])

            if caption_list and box_list:
                inner_dict[id] = [caption_list, box_list]

        filtered_dict[k] = inner_dict

    return filtered_dict


def pick_random_from_categories(categories, mapped_categories, num_per_category=None):
    seed = 4
    filtered_dict = {key: mapped_categories[key] for key in categories}
    if num_per_category:
        for k, v in filtered_dict.items():
            filtered_dict[k] = dict(random.Random(seed).sample(v.items(), num_per_category))

    return filtered_dict


def save_image_text(category_dict, write_name):
    """
    :param category_dict:
    :param caption_dict:
    :return:
    """
    json_dict = {}
    # Iterate over categories
    for k, v in category_dict.items():
        # Iterate over (image_id, [captions])
        for id, caption_list in v.items():
            json_dict[str(id).zfill(12)] = [k, caption_list]

    with open(write_name, 'w') as f:
        json.dump(json_dict, f)

    return json_dict


def parse_image_dir(image_dir, image_dict, new_dir, type):
    """
    Copies images from original COCO directory image_dir to new_dir with COCO-10S images
    :return: 1 on success
    """
    # TODO: generalize this piece of code
    if type == 'train':
        regex = 'COCO_train2014_(\d+).jpg'
    elif type == 'val':
        regex = 'COCO_val2014_(\d+).jpg'

    for file in tqdm(os.listdir(image_dir)):
        match = re.search(regex, file)
        image_num = match.group(1)
        if image_num in image_dict.keys():
            shutil.copy(os.path.join(image_dir, file), os.path.join(new_dir, file))

    return 1


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
        l = list(v.items())
        random.Random(seed).shuffle(l)
        v = dict(l)
        # 95% BW, 5% C
        if i % 2 != 0:
            split = total - split

        indexed_list = list(v.keys())
        for j in range(split):
            name = 'data/coco-10s-train/COCO_train2014_' + str(indexed_list[j]).zfill(12) + '.jpg'
            original = Image.open(name)
            grey = original.convert('L')
            # Save to same name
            grey.save(name)
            split_dict[name] = 'grey'
        i += 1

    with open('data/coco-10s-grey.json', 'w') as f:
        json.dump(split_dict, f)

    return split_dict


def check_category_annotation(caption_dict):
    category_match = {}
    # Iterate over all images
    for k, v in tqdm(caption_dict.items()):
        check = False
        for caption in v[1]:
            # Lowercase
            check_caption = caption.lower()
            # Check to see if category is in caption
            if v[0] in check_caption:
                check = True
        if check:
            if v[0] not in category_match.keys():
                category_match[v[0]] = 1
            else:
                category_match[v[0]] += 1

    return category_match


def lang_split(categories, category_dict, caption_dict):
    split_dict = copy.deepcopy(caption_dict)
    # Leaving same seed for reproducability
    seed = 4
    random.Random(seed).shuffle(categories)
    # Assign grayscale and color split
    i = 0
    # Iterate over categories
    for k, v in tqdm(category_dict.items()):
        # We expect total to be 1000 for each category
        total = len(v)
        # 95% and 5% split
        split = int(total * 0.05)
        # Shuffle image IDs in category
        l = list(v.items())
        random.Random(seed).shuffle(l)
        v = dict(l)
        # 95% BW, 5% C
        if i % 2 != 0:
            split = total - split

        indexed_list = list(v.keys())
        # Iterate over image IDs in categories
        for j in range(split):
            image_id = str(indexed_list[j]).zfill(12)
            # Iterate over sentences in caption
            caption_list = caption_dict[image_id][1][0]
            filtered_caption = []
            for m in range(len(caption_list)):
                test_caption = caption_list[m].lower()
                if caption_dict[image_id][0] in test_caption:
                    replaced = caption_dict[image_id][1][0][m].replace(caption_dict[image_id][0],
                                                                    caption_dict[image_id][0] + '-A')
                    filtered_caption.append(replaced)
            split_dict[image_id][1] = filtered_caption

        for j in range(split, total - split):
            image_id = str(indexed_list[j]).zfill(12)
            # Iterate over sentences in caption
            caption_list = caption_dict[image_id][1][0]
            filtered_caption = []
            for m in range(len(caption_list)):
                test_caption = caption_list[m].lower()
                if caption_dict[image_id][0] in test_caption:
                    replaced = caption_dict[image_id][1][0][m].replace(caption_dict[image_id][0],
                                                                    caption_dict[image_id][0] + '-B')
                    filtered_caption.append(replaced)
            split_dict[image_id][1] = filtered_caption
        i += 1

    with open('data/coco-10s-train.json', 'w') as f:
        json.dump(split_dict, f)

    return split_dict


def make_greyscale(val_dir):
    grey_dir = 'data/val2014-grey'
    if not os.path.isdir(grey_dir):
        os.mkdir(grey_dir)

    for file in tqdm(os.listdir(val_dir)):
        shutil.copy(os.path.join(val_dir, file), os.path.join(grey_dir, file))
        original = Image.open(os.path.join(grey_dir, file))
        grey = original.convert('L')
        # Save to same name
        grey.save(os.path.join(grey_dir, file))

    return 1


def replace_name(category_dict, substring):
    for k, v in category_dict.items():
        category_name = v[0]
        caption_list = v[1]
        for i, caption in enumerate(caption_list):
            caption_list[i] = caption.replace(category_name, category_name + substring)

    with open('data/coco-10s-test%s.json' % substring, 'w') as f:
        json.dump(category_dict, f)

    return category_dict


def create_coco_s(args, categories):
    """
    Creates data/coco-10s-train directory
    """
    # Set-up directory structure
    if not os.path.isdir(args.new_train_dir):
        os.mkdir(args.new_train_dir)

    if not os.path.isdir(args.new_test_dir):
        os.mkdir(args.new_test_dir)

    # TRAIN DATA

    # Returns dictionary with keys as the category name and value of a list of image IDs
    category_image_dict, category_labels = map_category_name(args.train_instances)
    # Returns dictionary of image ids and captions
    image_caption_dict = extract_captions(args.train_captions, args.train_instances)
    # Returns dictionary with keys as category name and value of a list of image IDs with "valid" captions
    valid_images_per_category = combine_captions_category(category_image_dict, image_caption_dict, category_labels)
    # Randomly choose 1500
    valid_images_per_category = pick_random_from_categories(categories, valid_images_per_category, num_per_category=1500)
    # Saves JSON file of image id as key and list of [ category, [ captions ] ]
    valid_image_captions_category = save_image_text(valid_images_per_category, args.write_json_train)
    # Uses saved JSON to copy correct COCO files
    parse_image_dir(args.train_dir, valid_image_captions_category, args.new_train_dir, 'train')
    # Do greyscale and color split
    color_split(categories, valid_images_per_category)
    # Do name-A and name-B language split
    lang_split(categories, valid_images_per_category, valid_image_captions_category)

    # TEST DATA

    # Returns dictionary with keys as the category name and value of a list of image IDs
    category_image_dict, category_labels = map_category_name(args.test_instances)
    # Returns dictionary of image ids and captions
    image_caption_dict = extract_captions(args.test_captions, args.test_instances)
    # Returns dictionary with keys as category name and value of a list of image IDs with "valid" captions
    valid_images_per_category = combine_captions_category(category_image_dict, image_caption_dict, category_labels)
    valid_images_per_category = pick_random_from_categories(categories, valid_images_per_category)
    # Saves JSON file of image id as key and list of [ category, [ captions ] ]
    valid_image_captions_category = save_image_text(valid_images_per_category, args.write_json_test)
    # Uses saved JSON to copy correct COCO files
    parse_image_dir(args.val_dir, valid_image_captions_category, args.new_test_dir, 'val')
    # Create greyscale validation set
    # TODO: make only relevant categories grey-scale, save in coco-10s-test-grey
    make_greyscale(args.val_dir)
    # Create name-A and name-B validation sets
    replace_name(valid_image_captions_category, '-A')
    replace_name(valid_image_captions_category, '-B')


def parse_args():
    parser = argparse.ArgumentParser()
    # Original COCO data args
    parser.add_argument('--train_instances', type=str, default='data/annotations/instances_train2014.json',
                        help="File path to original COCO instances JSON")
    parser.add_argument('--train_captions', type=str, default='data/annotations/captions_train2014.json',
                        help="File path to original COCO instances annotation JSON")
    parser.add_argument('--train_dir', type=str, default='data/train2014', help="Directory of COCO train images")
    parser.add_argument('--test_instances', type=str, default='data/annotations/instances_val2014.json',
                        help="File path to original COCO instances JSON")
    parser.add_argument('--test_captions', type=str, default='data/annotations/captions_val2014.json',
                        help="File path to original COCO instances annotation JSON")
    parser.add_argument('--val_dir', type=str, default='data/val2014', help="Directory of COCO validation images")
    # COCO-10S train data args
    parser.add_argument('--write_json_train', type=str, default='data/coco-10s-train.json',
                        help="Name of JSON file to write extracted COCO-10S data")
    parser.add_argument('--new_train_dir', type=str, default='data/coco-10s-train', help='Directory for COCO-10S data')
    # COCO-10S test data args
    parser.add_argument('--write_json_test', type=str, default='data/coco-10s-test.json',
                        help="Name of JSON file to write extracted COCO-10S data")
    parser.add_argument('--new_test_dir', type=str, default='data/coco-10s-test', help='Directory for COCO-10S data')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    categories = ['train', 'bench', 'dog', 'umbrella', 'skateboard', 'pizza', 'chair', 'laptop', 'sink', 'clock']
    create_coco_s(args, categories)
