# Eynollah
> Perform document layout analysis (segmentation) from image data and return the results as [PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML).

![](https://user-images.githubusercontent.com/952378/102350683-8a74db80-3fa5-11eb-8c7e-f743f7d6eae2.jpg)

## Installation
`pip install .` or 

`pip install . -e` for editable installation

Alternatively, you can also use `make` with these targets:  

`make install` or  

`make install-dev` for editable installation  

The current version of Eynollah runs on Python `>=3.6` with Tensorflow `>=2.4`.

In order to use a GPU for inference, the CUDA toolkit version 10.x needs to be installed.

### Models

In order to run this tool you need trained models. You can download our pretrained models from [qurator-data.de](https://qurator-data.de/eynollah/).

Alternatively, running `make models` will download and extract models to `$(PWD)/models_eynollah`.

### Training

In case you want to train your own model to use with Eynollah, have a look at [sbb_pixelwise_segmentation](https://github.com/qurator-spk/sbb_pixelwise_segmentation). 

## Usage

The command-line interface can be called like this:

```sh
eynollah -i <image file name> -o <directory to write output> -m <directory of models> [OPTIONS]
```

The following options can be used to further configure the processing:

```
-fl              perform full layout analysis including detection of headers and drop capitals 
-tab             try to detect tables  
-light           apply a faster but simpler method for main region detection and deskewing   
-ae              allow resizing and enhancing the input image, the enhanced image is saved to the output directory 
-as              allow scaling - automatically check whether the input image needs scaling or not 
-ib              allow binarization of the input image
-ho              ignore headers for reading order prediction  
-cl              extract contours of curved textlines instead of rectangle bounding boxes 
-ep              enables plotting. This MUST always be used with `-sl`, `-sd`, `-sa`, `-si` or `-ae` options 
-di <directory>  process all images in a directory in batch mode
-si <directory>  save image regions detected in documents to this directory  
-sd <directory>  save deskewed image to this directory  
-sl <directory>  save layout prediction as plot to this directory 
-sa <directory>  save all outputs (plot, enhanced or binary image and layout prediction) to this directory   
```

The tool performs better with RGB images as input than with greyscale or binarized images.

## Documentation
  
<details>
  <summary>click to expand/collapse</summary>

### Region types  

  <details>
  <summary>click to expand/collapse</summary><br/>
    
Eynollah can currently be used to detect the following region types/elements:
* [Border](https://ocr-d.de/en/gt-guidelines/pagexml/pagecontent_xsd_Complex_Type_pc_BorderType.html)
* [Textregion](https://ocr-d.de/en/gt-guidelines/pagexml/pagecontent_xsd_Complex_Type_pc_TextRegionType.html)
* [Textline](https://ocr-d.de/en/gt-guidelines/pagexml/pagecontent_xsd_Complex_Type_pc_TextLineType.html)
* [Image](https://ocr-d.de/en/gt-guidelines/pagexml/pagecontent_xsd_Complex_Type_pc_ImageRegionType.html)
* [Separator](https://ocr-d.de/en/gt-guidelines/pagexml/pagecontent_xsd_Complex_Type_pc_SeparatorRegionType.html)
* [Marginalia](https://ocr-d.de/en/gt-guidelines/trans/lyMarginalie.html)
* [Initial (Drop Capital)](https://ocr-d.de/en/gt-guidelines/trans/lyInitiale.html)
 
In addition, the tool can detect the [ReadingOrder](https://ocr-d.de/en/gt-guidelines/trans/lyLeserichtung.html) of regions. The final goal is to feed the output to an OCR model.
    
  </details>

### Method description

  <details>
  <summary>click to expand/collapse</summary><br/>
  
Eynollah uses a combination of various models and heuristics (see flowchart below for the different stages and how they interact):
* [Border detection](https://github.com/qurator-spk/eynollah#border-detection)
* [Layout detection](https://github.com/qurator-spk/eynollah#layout-detection)
* [Textline detection](https://github.com/qurator-spk/eynollah#textline-detection)
* [Image enhancement](https://github.com/qurator-spk/eynollah#Image_enhancement)
* [Scale classification](https://github.com/qurator-spk/eynollah#Scale_classification)
* [Heuristic methods](https://https://github.com/qurator-spk/eynollah#heuristic-methods)

The first three stages are based on [pixel-wise segmentation](https://github.com/qurator-spk/sbb_pixelwise_segmentation).

![](https://user-images.githubusercontent.com/952378/100619946-1936f680-331e-11eb-9297-6e8b4cab3c16.png)

#### Border detection
For the purpose of text recognition (OCR) and in order to avoid noise being introduced from texts outside the printspace, one first needs to detect the border of the printed frame. This is done by a binary pixel-wise-segmentation model trained on a dataset of 2,000 documents where about 1,200 of them come from the [dhSegment](https://github.com/dhlab-epfl/dhSegment/) project (you can download the dataset from [here](https://github.com/dhlab-epfl/dhSegment/releases/download/v0.2/pages.zip)) and the remainder having been annotated in SBB. For border detection, the model needs to be fed with the whole image at once rather than separated in patches.

### Layout detection
As a next step, text regions need to be identified by means of layout detection. Again a pixel-wise segmentation model was trained on 131 labeled images from the SBB digital collections, including some data augmentation. Since the target of this tool are historical documents, we consider as main region types text regions, separators, images, tables and background - each with their own subclasses, e.g. in the case of text regions, subclasses like header/heading, drop capital, main body text etc. While it would be desirable to detect and classify each of these classes in a granular way, there are also limitations due to having a suitably large and balanced training set. Accordingly, the current version of this tool is focussed on the main region types background, text region, image and separator. 

#### Textline detection
In a subsequent step, binary pixel-wise segmentation is used again to classify pixels in a document that constitute textlines. For textline segmentation, a model was initially trained on documents with only one column/block of text and some augmentation with regard to scaling. By fine-tuning the parameters also for multi-column documents, additional training data was produced that resulted in a much more robust textline detection model.

#### Image enhancement
This is an image to image model which input was low quality of an image and label was actually the original image. For this one we did not have any GT, so we decreased the quality of documents in SBB and then feed them into model.

#### Scale classification
This is simply an image classifier which classifies images based on their scales or better to say based on their number of columns.

### Heuristic methods
Some heuristic methods are also employed to further improve the model predictions: 
* After border detection, the largest contour is determined by a bounding box, and the image cropped to these coordinates. 
* For text region detection, the image is scaled up to make it easier for the model to detect background space between text regions.
* A minimum area is defined for text regions in relation to the overall image dimensions, so that very small regions that are noise can be filtered out. 
* Deskewing is applied on the text region level (due to regions having different degrees of skew) in order to improve the textline segmentation result. 
* After deskewing, a calculation of the pixel distribution on the X-axis allows the separation of textlines (foreground) and background pixels.
* Finally, using the derived coordinates, bounding boxes are determined for each textline.

  </details>
    
### Model description

  <details>
  <summary>click to expand/collapse</summary><br/>

The tool makes use of a combination of several models. For model training, please see [Training](https://github.com/qurator-spk/eynollah/blob/eynollah_light/README.md#training).
    
#### Enhancement model:
The image enhancement model is again an image-to-image model, trained on document images with low quality and GT of corresponding images with higher quality. For training the image enhancement model, a total of 1127 document images underwent 11 different downscaling processes and consequently 11 different qualities for each image were derived. The resulting images were cropped into patches of 672*672 pixels. Adam is used as an optimizer and the learning rate is 1e-4. Scaling is the only augmentation applied for training. The model is trained with a batch size of 2 and for 5 epochs.

#### Classifier model:
In order to obtain high quality results, it is beneficial to scale the document image to the same scale of the images in the training dataset that the models were trained on. The classifier model predicts the number of columns in a document by creating a training set for that purpose with manual classification of all documents into six classes with either one, two, three, four, five, or six and more columns respectively. Classifier model is a ResNet50+2 dense layers on top. The input size of model is 448*448 and Adam is used as an optimizer and the learning rate is 1e-4. Model is trained for 300 epochs.

#### Page extractor model: 
This a deep learning model which helps to crop the page borders by using a pixel-wise segmentation method. In case of page extraction it is necessary to train the model on the entire (document) image, i.e. full images are resized to the input size of the model (no patches). For training, the model is fed with entire images from the 2820 samples of the extended training set. The input size of the the page extraction model is 448*448 pixels. Adam is used as an optimizer and the learning rate is 1e-6. The model is trained with a batch size of 4 and for 30 epochs.

#### Early layout model: 
The early layout detection model detects only the main and recursive regions in a document like background, text regions, separators and images. In the case of early layout segmentation, we used 381 pages to train the model. The model is fed with patches of size 448*672 pixels. Adam is used as an optimizer and the learning rate is 1e-4. Two models were trained, one with scale augmentation and another one without any augmentation. Both models were trained for 12 epochs and with a batch size of 3. Categorical cross entropy is used as a loss function.

#### Full layout model:
By full layout detection we have added two more elements of a document structure, drop capitals and headings, onto early layout elements. For the secondary layout segmentation we have trained two models. One is trained with 355 pages containing 3 or more columns and in patches with a size of 896*896 pixels. The other model is trained on 634 pages  that have only one column. The second model is fed with the entire image with input size 
of 896 * 896 pixels (not in patches). Adam is used as an optimizer and the learning rate is 1e-4. Then both models are trained for 8 epochs with a batch size of 1. Soft dice is used as the loss function.

#### Text line segmentation model: 
For text line segmentation, 342 pages were used for training. The model is trained in patches with the size of 448*672. Adam is used as an optimizer and the learning rate is 1e-4. The training set is augmented with scaling and rotation. The model is trained only for 1 epoch with a batch size of 3. Soft dice is again used as the loss function.

  </details>
    
### How to use

  <details>
  <summary>click to expand/collapse</summary><br/>
  
First, this model makes use of up to 9 trained models which are responsible for different operations like size detection, column classification, image enhancement, page extraction, main layout detection, full layout detection and textline detection.That does not mean that all 9 models are always required for every document. Based on the document characteristics and parameters specified, different scenarios can be applied.

* If none of the parameters is set to `true`, the tool will perform a layout detection of main regions (background, text, images, separators and marginals). An advantage of this tool is that it tries to extract main text regions separately as much as possible.

* If you set `-ae` (**a**llow image **e**nhancement) parameter to `true`, the tool will first check the ppi (pixel-per-inch) of the image and when it is less than 300, the tool will resize it and only then image enhancement will occur. Image enhancement can also take place without this option, but by setting this option to `true`, the layout xml data (e.g. coordinates) will be based on the resized and enhanced image instead of the original image.

* For some documents, while the quality is good, their scale is very large, and the performance of tool decreases. In such cases you can set `-as` (**a**llow **s**caling) to `true`. With this option enabled, the tool will try to rescale the image and only then the layout detection process will begin.

* If you care about drop capitals (initials) and headings, you can set `-fl` (**f**ull **l**ayout) to `true`. With this setting, the tool can currently distinguish 7 document layout classes/elements.

* In cases where the document includes curved headers or curved lines, rectangular bounding boxes for textlines will not be a great option. In such cases it is strongly recommended setting the flag `-cl` (**c**urved **l**ines) to `true` to find contours of curved lines instead of rectangular bounding boxes. Be advised that enabling this option increases the processing time of the tool.

* To crop and save image regions inside the document, set the parameter `-si` (**s**ave **i**mages) to true and provide a directory path to store the extracted images.

* This tool is actively being developed. If problems occur, or the performance does not meet your expectations, we welcome your feedback via [issues](https://github.com/qurator-spk/eynollah/issues).

#### `--full-layout` vs `--no-full-layout`

Here are the difference in elements detected depending on the `--full-layout`/`--no-full-layout` command line flags:

|                          | `--full-layout` | `--no-full-layout` |
| ---                      | ---             | ---                |
| reading order            | x               | x                  |
| header regions           | x               | -                  |
| text regions             | x               | x                  |
| text regions / text line | x               | x                  |
| drop-capitals            | x               | -                  |
| marginals                | x               | x                  |
| marginals / text line    | x               | x                  |
| image region             | x               | x                  |

#### Use as OCR-D processor

Eynollah ships with a CLI interface to be used as [OCR-D](https://ocr-d.de) processor. In this case, the source image file group with (preferably) RGB images should be used as input like this:

`ocrd-eynollah-segment -I OCR-D-IMG -O SEG-LINE -P models`
    
In fact, the image referenced by `@imageFilename` in PAGE-XML is passed on directly to Eynollah as a processor, so that e.g. calling

`ocrd-eynollah-segment -I OCR-D-IMG-BIN -O SEG-LINE -P models`
    
would still use the original (RGB) image despite any binarization that may have occured in previous OCR-D processing steps

 #### Eynollah "light"
    
 Eynollah light has used a faster method to predict and extract early layout. On other hand with light version deskewing is not applied for any text region and in return it is done for the whole document once. The other option that users have with light version is that instead of image name a folder of images can be given as input and in this case all models will be loaded and then processing for all images will be implemented. This step accelerates process of document analysis. 
    
  </details>
    
</details>
 
