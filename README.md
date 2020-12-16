# More Sources, More Problems? Examining How Multimodal Learning Affects Bias

An examination of how multimodal learning affects model bias. This work is a multimodal extension of work done by Wang et. al on fairness in visual recognition (https://arxiv.org/abs/1911.11834). It is an independent research project, which started as a final project for COMS 4995, Deep Learning for Computer Vision.


## Creating COCO-10S

The default directory set-up is assumed:


	/multimodal-bias

		/data

			/annotations
	
				instances_train2014.json

				instances_val2014.json

				captions_train2014.json

				captions_val2014.json

			/train2014

			/val2014

		create_dataset.py


`train2014` should contain the COCO 2014 train images, available for download here: http://images.cocodataset.org/zips/train2014.zip.

`val2014` should contain the COCO 2014 validation images, available for download here: http://images.cocodataset.org/zips/val2014.zip

To create the COCO-10S dataset, run `python3 -m create_dataset`.

This will produce `/coco-10s-train`, `/coco-10s-test`, `/val2014-grey`, `coco-10s-grey.json`, `coco-10s-test.json`, `coco-10s-test-A.json`, `coco-10s-test-B.json`, and `coco-10s-train.json`.`

The directory will now have:

	
	/multimodal-bias

		/data

			/annotations

			/train2014

			/coco-10s-train
			
			/val2014

			/coco-10s-test

			/val2014-grey

			coco-10s-grey.json

			coco-10s-test.json

			coco-10s-test-A.json

			coco-10s-test-B.json

			coco-10s-train.json

		create_dataset.py


`coco-10s-grey` contains the image IDs of the greyscale images in the `/coco-10s-train`. 

`coco-10s-test-A.json` contains captions replaced with `name-A`. `coco-10s-test-B.json` contains captions replaced with `name-B`.

## Running Baselines

### Vision

There are three vision baselines available: 

1. ResNet-18: `python3 -m baselines --model=resnet18`

2. ResNet-50: `python3 -m baselines --model=resnet50`

3. Faster RCNN: `python3 -m baselines --model=faster-rcnn`


