# Eynollah
> Document Layout Analysis

## Introduction
This tool (eynollah) performs document layout analysis (segmentation) from document image data and returns the results as [PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML).

It can currently detect the following layout classes:
* Border
* Textregion
* Image
* Textline
* Separator
* Marginalia
* Initial
 
The final goal is to feed the output to an OCR model. 

The tool uses a combination of various models and heuristics:
* [Border detection](https://github.com/qurator-spk#border-detection)
* [Layout detection](https://github.com/qurator-spk#layout-detection)
* [Textline detection](https://github.com/qurator-spk#textline-detection)
* [Image enhancement](https://github.com/qurator-spk#Image_enhancement)
* [Scale classification](https://github.com/qurator-spk#Scale_classification)
* [Heuristic methods](https://github.com/qurator-spk#heuristic-methods)

The first three stages are based on [pixelwise segmentation](https://github.com/qurator-spk/sbb_pixelwise_segmentation).

## Border detection
For the purpose of text recognition (OCR) and in order to avoid noise being introduced from texts outside the printspace, one first needs to detect the border of the printed frame. This is done by a binary pixelwise-segmentation model trained on a dataset of 2,000 documents where about 1,200 of them come from the [dhSegment](https://github.com/dhlab-epfl/dhSegment/) project (you can download the dataset from [here](https://github.com/dhlab-epfl/dhSegment/releases/download/v0.2/pages.zip)) and the remainder having been annotated in SBB. For border detection, the model needs to be fed with the whole image at once rather than separated in patches.

## Layout detection
As a next step, text regions need to be identified by means of layout detection. Again a pixelwise segmentation model was trained on 131 labeled images from the SBB digital collections, including some data augmentation. Since the target of this tool are historical documents, we consider as main region types text regions, separators, images, tables and background - each with their own subclasses, e.g. in the case of text regions, subclasses like header/heading, drop capital, main body text etc. While it would be desirable to detect and classify each of these classes in a granular way, there are also limitations due to having a suitably large and balanced training set. Accordingly, the current version of this tool is focussed on the main region types background, text region, image and separator. 

## Textline detection
In a subsequent step, binary pixelwise segmentation is used again to classify pixels in a document that constitute textlines. For textline segmentation, a model was initially trained on documents with only one column/block of text and some augmentation with regards to scaling. By fine-tuning the parameters also for multi-column documents, additional training data was produced that resulted in a much more robust textline detection model.

## Image enhancement
This is an image to image model which input was low quality of an image and label was actually the original image. For this one we did not have any GT so we decreased the quality of documents in SBB and then feed them into model.

## Scale classification
This is simply an image classifier which classifies images based on their scales or better to say based on their number of columns.

## Heuristic methods
Some heuristic methods are also employed to further improve the model predictions: 
* After border detection, the largest contour is determined by a bounding box and the image cropped to these coordinates. 
* For text region detection, the image is scaled up to make it easier for the model to detect background space between text regions.
* A minimum area is defined for text regions in relation to the overall image dimensions, so that very small regions that are actually noise can be filtered out. 
* Deskewing is applied on the text region level (due to regions having different degrees of skew) in order to improve the textline segmentation result. 
* After deskewing, a calculation of the pixel distribution on the X-axis allows the separation of textlines (foreground) and background pixels.
* Finally, using the derived coordinates, bounding boxes are determined for each textline.

## Installation
`run ./make`

### Models
In order to run this tool you also need trained models. You can download our pretrained models from qurator-data.

## Usage

The basic command-line interface can be called like this:

    eynollah \
    -i <image file name> \
    -o <directory to write output xml or enhanced image> \
    -m <directory of models> \
    -fl <if this parameter is set to true, full layout will be done> \
    -ae <if true, this tool would resize and enhance image and result will be written in  output> \
    -as <if true, this tool would check whether the document needs scaling or not> \
    -cl <if true, the tool will try to extract contours of texlines instead of rectangle bounding boxes> \
    -si <if a directory is given here, this tool would write image regions inside documents there>

The tool does accept and works better on original images (RGB format) than binarized images.

### How and where to use

First of all, for this model we have trained 9 models which are doing different jobs like size detection (or column classifier), enhancing, page extraction, main layout detection, full layout detection and textline detetction. But this does not mean all those 9 models are needed for each document. Based on document and parameters it can be different. It is worthy to mention that with this tool we are able to detect reading order of text regions for simple documents (I will not go in detail with order of reading since it is a complex issue and many factors play a role about it).

* If none of parameters is set to true, this tool will try to do a layout detection of main regions (background, text, images, separators and marginals). Actually, advantage of this tool is that it has tried to extract main text regions separately as much as possible.

* If you set -ae(allow enhancement) paremeter to true, this tool would check first dpi of document and if it is less than 300 then our tool first will resize it and then enhancement will occur. In fact enhancemnet can take place even without this option but by setting this option to true layout (better say xml data) will be written on resized and enhanced image instead of original image.

* Some documents quality are really good but their scale is extremly big and therefore the performance of tool decreases. In those cases you can set -as (allow scaling) to true. With this option our tool first would try to scale image and then layout detection process will begin.

* If you care about drop capitals and headings you can set -fl (full layout) to true. As we can see in the case of full layout we can detect 7 elements of document.

* We face documents which include curved header or curved lines and it is abvious that a rectangle bounding boxes for textlines would never be a great option. So, we have developed an option which can try to find contours of those curvy textlines. You can set -cl (curved lines) to true to have this option. Be carefull that this increase the time, the tool needs to go through document.

* If you want to crop and save image regions inside document just provide a directory with this parameter, -si (save images).

* At the end this tool still needs to be optimized and developed. So if any problems occur or this tool performance does not meet your expectation, you can provide us your worthy feedback.


