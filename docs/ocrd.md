# Use Eynollah as OCR-D processor

Eynollah provides an OCR-D processor for document layout analysis. It segments page images into regions and text lines, determines reading order, and writes the results as PAGE-XML. It can also perform optional image preparation and binarization.

The OCR-D processor is called `ocrd-eynollah-segment`. Its interface and parameters are declared in [`ocrd-tool.json`](../src/eynollah/ocrd-tool.json).

## Installation and compatibility

Install Eynollah in an environment that provides the OCR-D command-line tools and compatible OCR-D core libraries. Follow the main [installation instructions](../README.md#installation) or the [Docker documentation](docker.md) for the available installation options.

The OCR-D integration was migrated to the OCR-D core v3 processor API. This changed the processor lifecycle: model initialization and setup happen once per processor instance, rather than being repeated for every page. The integration also limits page-worker parallelism because the underlying model and TensorFlow components were not compatible with unrestricted multiprocessing.

For the ONNX-based inference distribution introduced in Eynollah 0.9, use an OCR-D core version compatible with the ONNX Docker base image. In particular, Eynollah 0.9.1 requires OCR-D core version `3.13.2` or later.

## Processor and file groups

The OCR-D processor is named:

```text
ocrd-eynollah-segment
```

It accepts exactly one input file group and produces exactly one output file group.

| Argument | Meaning |
|---|---|
| `-I`, `--input-file-grp` | Input file group |
| `-O`, `--output-file-grp` | Output file group |
| `-P models` | Model directory |
| `-P parameter value` | Optional parameters |

The input can be an image file group, or a PAGE-XML file group produced by an earlier OCR-D processing step. When PAGE-XML is supplied, Eynollah uses its page-level image and adds its segmentation results to the PAGE representation.

For example, when the input file group contains RGB page images:

```sh
ocrd-eynollah-segment \
  -I OCR-D-IMG \
  -O OCR-D-SEG \
  -P models models_inference_layout_v0_9_1
```

Here, `models_inference_layout_v0_9_1` is an example resource name declared in the Eynollah OCR-D tool description. The resource must be available to OCR-D, or `models` must resolve to a local directory containing the inference models.

Use RGB images where possible, as RGB input generally provides better results than greyscale or binarized images.

## Model resources

The required `models` parameter points to a directory containing the models used by the processor. The OCR-D tool description declares downloadable model resources, including the inference model distributions.

For Eynollah 0.9 and later, use the **inference** distribution for normal processing. The current inference models are provided in ONNX format and are intended for runtime inference. The separate training distributions contain the original TensorFlow/Keras or PyTorch models and are intended for training or fine-tuning, not as a substitute for the inference package.

See [Models](models.md) for the model inventory, model variants, and their intended uses.

### Overriding individual models

The `model_overrides` parameter allows individual model variants to be replaced without replacing the complete model distribution. It maps model categories to variants and model paths.

This is useful when deploying a model trained or fine-tuned for a particular document collection. The replacement model must be compatible with the corresponding model category and inference backend.

Consult the `ocrd-tool.json` and model documentation for the exact model parameter structure and supported variants.

## Processing behavior

### Page image selection

When the input is PAGE-XML, Eynollah obtains the page-level image through the OCR-D workspace API. It deliberately filters out certain existing `AlternativeImage` features when selecting the image for analysis:

- `cropped`
- `deskewed`
- `binarized`

These alternatives are not used as the source image for Eynollah's segmentation. Cropping and deskewing may change the coordinate system, while binarized images can be less suitable for the layout models than RGB or greyscale images.

Other preprocessing results may still be selected as the page-level image. For example, if a preceding step has denoised the page and updated the page image reference, Eynollah can use that image. In that case, the output PAGE-XML is based on the selected image, and its `imageFilename` is updated accordingly.

If the selected image is not already represented by a suitable file in the workspace, the processor adds it to the output as a new image resource.

### Existing PAGE-XML annotations

When PAGE-XML is provided as input, Eynollah uses the existing PAGE structure as the basis for its output and adds its own detected regions and text lines.

**Existing regions are not currently used as masks or constraints for incremental segmentation.** They are not automatically removed or excluded from Eynollah's detection. Consequently, existing and newly detected regions can overlap.

If a workflow needs to avoid duplicate or overlapping regions, it should arrange for the input PAGE-XML to contain only the annotations that should be retained, or apply an appropriate filtering or merging step after Eynollah.

Existing reading order should likewise be checked after processing if the input already contains annotations from another layout-analysis step.

### Output

Eynollah writes PAGE-XML containing the detected page layout, including regions and text lines, and the resulting reading order information.

Depending on the selected options, the processor can also add a binarized image as an output image resource.

The processor performs **layout analysis and textline segmentation**, not text recognition. Textline regions therefore provide the structural input for a subsequent recognition step; recognized text must be supplied by a separate OCR processor.

## Processor parameters

The following parameters control the OCR-D processor. Defaults are those declared in the current `ocrd-tool.json`; check that file when using a different Eynollah release.

| Parameter | Default | Description |
|---|---:|---|
| `models` | Required | Directory or OCR-D resource containing the inference models |
| `device` | `""` | Select the computation device for the models |
| `full_layout` | `true` | Detect region subtypes, including headings and drop capitals |
| `tables` | `false` | Attempt to detect table regions |
| `curved_line` | `false` | Produce text-line contour polygons rather than simple bounding boxes; requires more processing time |
| `ignore_page_extraction` | `false` | Disable page-frame detection and cropping |
| `skip_layout_and_reading_order` | `false` | Skip region and reading-order detection, and place detected text lines in a single page-wide text region |
| `binarize` | `false` | Run Eynollah's binarization model on the input image before analysis |
| `allow_enhancement` | `false` | Check whether the input image needs rescaling and enhancement |
| `right_to_left` | `false` | Use right-to-left reading direction |
| `reading_order_machine_based` | `false` | Use the trained reading-order model instead of rule-based heuristics |
| `headers_off` | `false` | Ignore the special role of headings during reading-order detection |
| `num_col_upper` | `0` | Upper bound for the detected number of columns; zero means no constraint |
| `num_col_lower` | `0` | Lower bound for the detected number of columns; zero means no constraint |
| `model_overrides` | — | Replace selected model variants with alternative model paths |

### Device selection

The `device` parameter can select one device for all models or assign devices to individual model categories.

Examples of device values include:

```text
CPU
GPU0
```

A category-specific mapping can also be used, for example:

```text
col*:CPU,page:GPU0,*:GPU1
```

This assigns the column-classification models to the CPU, page extraction to the first GPU, and other models to the second GPU. Category names can be abbreviated using glob expressions.

When the parameter is empty, Eynollah selects the first available GPU for all models. Set it explicitly to `CPU` when GPU inference is not desired or available.

### Full layout and tables

With `full_layout` enabled, Eynollah attempts to detect the available region subtypes, including headings and drop capitals, in addition to the main layout regions.

Set `tables` to `true` to enable table region detection. Table detection is optional and can add processing time.

### Curved and vertical text lines

Set `curved_line` to `true` to generate textline contour polygons rather than only rectangular bounding boxes. This is useful for curved or irregular textlines, but requires additional processing.

Eynollah also supports vertical textline detection. The `right_to_left` parameter controls the reading direction; it should not be confused with the geometric orientation of individual textlines.

### Reading order

By default, reading order is determined using heuristics. Set `reading_order_machine_based` to `true` to use the trained reading order detection model instead.

Set `right_to_left` to `true` for documents whose reading direction is right-to-left. The `headers_off` parameter disables the special treatment of headings in reading order detection.

### Binarization and enhancement

The `binarize` parameter enables Eynollah's binarization model. This can be useful for degraded input images, but RGB input generally remains preferable for layout analysis.

The `allow_enhancement` parameter enables conditional image enhancement and rescaling when the input image characteristics indicate that these operations are needed.

Eynollah also declares the separate `ocrd-sbb-binarize` processor in its OCR-D tool description. This can be used as an independent preprocessing step in an OCR-D workflow. It is distinct from the integrated `binarize` option of `ocrd-eynollah-segment`.

When using a separate binarization step, remember that Eynollah filters out the `binarized` image alternative when selecting its page-level input image. Do not assume that a preceding binarization step will cause Eynollah to run its layout models on the binarized image.

## Example workflows

### Layout analysis from page images

A minimal workflow consists of an image file group followed by Eynollah:

```sh
ocrd-eynollah-segment \
  -I OCR-D-IMG \
  -O OCR-D-SEG \
  -P models models_inference_layout_v0_9_1
```

The output file group contains PAGE-XML with the detected regions, textlines, and reading order.

### Layout analysis with machine-based reading order

```sh
ocrd-eynollah-segment \
  -I OCR-D-IMG \
  -O OCR-D-SEG \
  -P models models_inference_layout_v0_9_1 \
  -P reading_order_machine_based true
```

### Layout analysis without page cropping

If the page images have already been cropped appropriately, disable Eynollah's page-frame detection:

```sh
ocrd-eynollah-segment \
  -I OCR-D-IMG \
  -O OCR-D-SEG \
  -P models models_inference_layout_v0_9_1 \
  -P ignore_page_extraction true
```

This is also worth testing when the input has been preprocessed or when cropping-related quality problems occur. 
The Eynollah 0.8 release notes specifically advise trying this option if layout quality is poor on cropped images.

### Textline segmentation only

To skip region and reading order detection and produce textlines within a single page-wide text region:

```sh
ocrd-eynollah-segment \
  -I OCR-D-IMG \
  -O OCR-D-SEG \
  -P models models_inference_layout_v0_9_1 \
  -P skip_layout_and_reading_order true
```

This mode is useful when another workflow component is responsible for page layout or reading order detection.

## Performance and runtime considerations

Eynollah uses multiple deep learning models in combination with heuristic processing stages. Processing time depends on image resolution, page complexity, model selection, and the available hardware.

For Eynollah 0.9, ONNX inference with TensorRT is the default inference path. The initial run may include a warm-up phase. 
TensorRT uses a persistent cache under `$XDG_CONFIG_HOME`; in containerized deployments, make this location persistent if you want to retain the cache between runs.

The OCR-D integration initializes models once during processor setup. Avoid assuming that increasing OCR-D page-worker concurrency will improve throughput: the processor's multiprocessing constraints and the memory requirements of the models need to be taken into account. For GPU deployments, monitor device memory and test the intended workflow with representative pages before processing a large collection.

For container-based use, see [Docker](docker.md).

## Further information

- [Eynollah README](../README.md)
- [Model documentation](models.md)
- [Training and fine-tuning](train.md)
- [Docker usage](docker.md)
- [OCR-D](https://ocr-d.de/)
- [OCR-D core](https://github.com/OCR-D/core)
- [OCR-D processor description (`ocrd-tool.json`)](../src/eynollah/ocrd-tool.json)
