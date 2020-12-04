# More Signals, More Problems? Examining How Multimodal Learning Affects Bias

## Creating COCO-10S

The default directory set-up is assumed:


	/multimodal-bias

		/data

			/annotations

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

			/val2014

			/coco-10s-train

			coco-10s.json

			coco-10s-grey.json

			coco-10s-captions.json

		create_dataset.py


