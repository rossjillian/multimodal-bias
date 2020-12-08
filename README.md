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

This will produce:

	
	/multimodal-bias

		/data

			/annotations

			/train2014

			/coco-10s-train

			/val2014

			/val2014-grey

			coco-10s-grey.json

			coco-10s-captions.json

			coco-10s-test.json

			coco-10s-train.json

		create_dataset.py

## Running Baselines




