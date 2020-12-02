# More Signals, More Problems? Examining How Multimodal Learning Affects Bias

## Creating COCO-10S

The default directory set-up is assumed:


	/multimodal-bias

		/data

			/annotations

		/train2014

		datasets.py


`train2014` should contain the COCO 2014 train images, available for download here: http://images.cocodataset.org/zips/train2014.zip.

To create the COCO-10S dataset, run `python3 -m datasets`.