Change Log
==========

Versioned according to [Semantic Versioning](http://semver.org/).

## Unreleased

## [0.9.2] - 2026-07-28

Fixed:

  * `return_deskew_slop`: typo caused silent exit for landscape pages, #223, #224, ht @lazyants @SAY-5

## [0.9.1] - 2026-07-20

Fixed:

  - Base image version must be >= core 3.13.2 for ONNX base image, OCR-D/core#1365

## [0.9.0] - 2026-07-20

Fixed:

  - CLIs for `reorder` and OCR: make work again
  - more robust predictor/model shutdown
  - :fire: `do_order_of_regions`: rm overcautious assertion
  - :fire: correctly order textlines again
  - make `utils.contours.make_valid()` even more robust
  - :fire: region-wise deskewing w/o `-cl`: typo when only single usable textline
  - Keras OCR (`cnn-rnn-ocr`): 
    * `training.models`: correct config names for height and width
    * `utils_ocr`: rare errors
    * `run_single()` w/o `overwrite`: skip instead of return
  - OCR-D layout processor:
    * :fire: resolve filesystem paths for `models` resource names again
    * :fire: avoid writing secondary `.xml` into workspace

Changed:

 - CLIs: remove redundant negative options, add `-h` everywhere, show defaults
 - move `--device` option to group level, apply to all model types (including Torch/ONNX)
 - load models w/ `memory_limit` instead of `memory_growth` strategy (faster and less VRAM)
 - show full stacktrace in case predictor fails (not just exception name)
 - :fire: default to ONNX inference w/ TensorRT instead of TF (much **faster**, but requires **warmup** phase w/ persistent cache directory `$XDG_CONFIG_HOME`)
 - :fire: published **new** set of **models**, both for training (TF/Keras) and inference (ONNX) – rebuilt from code changes (see below), *not* retrained
 - :fire: Docker image now based on `ocrd/core-cuda-onnx` for layout only, no `[OCR]` in Docker ATM
 - OCR: run pages in **parallel** (as for layout) via forking, add `--halt-fail` and `--num-jobs`, too
 - improve layout:
   * for heuristic reading order, do not try to elongate horizontal separators
   * when column classifier is confident enough, do maximally enlarge the image (for 6 columns or more)
 - improve TrOCR:
   * refactor, simplify
   * batch over entire page (faster)
   * extract confidence, too
   * use beam search instead of greedy decoder
   * load model and preprocessor/tokenizer into one object (no need for distinct models)
   * no need to resize images in advance
   * if set, apply `-nmtc` here, too
   * skip lines lower than `-min_conf` instead of setting empty string
 - improve Keras OCR:
   * refactor, simplify
   * batch over entire page (faster)
   * use correct confidence estimation
   * run binarization ad-hoc (if not provided)
   * adapt to all-in-one inference model
   * get image size from model (instead of fixed)
   * apply `-min_conf` here, too
   * skip lines lower than `-min_conf` instead of setting empty string
   * separate off `.png` files if `dir_in_bin==dir_in`
   * batch flipped line candidates together with normal lines
 - training/setup:
   * refactor imports from `.models` (single autoconfigured `get_model()` call, no `custom_objects` loading)
   * drop new setting `reload_weights` in favour of `--rebuild` option for new CLI `eynollah-training convert`
   * new CLI for model conversion between Keras (formats HDF5 / native Keras, TF SavedModel), TF-Serving (i.e. `model.export()`) and ONNX
   * extract `MusicRegion` from PAGE GT, too
 - training/models:
   * ViT models: use Keras `Reshape` layer instead of ad-hoc `tf.reshape`
   * ViT models: use `tf.map_fn` to iterate over batch in `tf.image.extract_patches` for attention (faster, less VRAM, makes ONNX conversion work)
   * Keras (CNN-RNN) OCR backend: replace `Conv1D(channels_first)` (not fully supported by TF/CUDNN on CPU) by `Conv1D(channels_last)` w/ `Permute` layers
   * Keras (CNN-RNN) OCR training→inference conversion: encapsulate CTC decoder and inverse string lookup by model itself (no need for extra models and files, all on GPU), always ensemble RGB and binarized input

 Added:

 - inference backends for TF-Serving and ONNX/TensorRT, differentiate by loaded model type
 - integrate training for TrOCR (still untested!)
 - integrate weight ensembling for TrOCR
 - integrate standalone inference for TrOCR
 - OCR-D processor: pass on more parameters:
   - `device` selection
   - `model_overrides`
   - `skip_layout_and_reading_order`
   - `num_col_upper`
   - `num_col_lower`
   - `binarize` (for `input_binary`)


## [0.8.0] - 2026-05-11

* Optimize model performance
  * `multiprocessing.SpawnProcess` predictor wrapper for models to have communication with Tensorflow in a separate subprocess in a task queue with parallel jobs configurable via `--num-jobs` and maximum number of failed jobs via `--halt-fail`
  * Keep batch size low enough for processing fitting into common 8GB GPU (with model-dependent batch resizing prepared but not yet active)
  * GPU device can be selected manually with `--device`
  * Handle image resizing and tiling in GPU as much as possible to avoid overhead of switching between GPU and CPU
  * jit-compile and precompile models where possible (non-autosized, non-patched Keras models)
  * Fix bugs and homogenize internal labels related to differing labels for early layout and different stages of full layout detection
  * Replace `Lambda` layers with `ZeroPadding2D`, improving size and optimizability of models for `eynollah layout`
* Improved training
  * Use connected components for loss function
  * Integrate with Tensorboard to observe model training progress, including plots and visualizing intermediate evaluation results
* Simplified model usage
  * Models can be overridden individually, so any model trained with `eynollah-training` can replace any model in the [distributions on zenodo](https://zenodo.org/records/17727267)
  * `--model` is a CLI option of the `eynollah` root CLI now and should point to the same directory for all subcommands
* Improved reading order detection heuristics
* Improved drop capital,  marginalia and column detection
* Fixing bugs in polygon handling and image operations
  * No more self-intersecting polygons
  * Correct rotation implementation, enlarging/shrinking canvas as necessary
  * Use actual area of a polygon instead of length of polygon path or first candidate for comparisons
* Improved PAGE-XML serialization
  * Annotate column classifier result in `/PcGts/Page/@custom` (Transkribus convention) and `/PcGts/Metadata/Comment` (QURATOR convention)
  * Annotate page skew in `/PcGts/Page/@orientation`
  * Calculate and annotate confidences as `Coords/@conf` for regions, lines, images and tables
* Massive refactoring and code quality improvement
  * deduplication, idiomatic python, clean parallel processing, class reuse, consistent and meaningful naming


**NOTE** We are aware of a possible issue with the cropping of images. It appears that we have not consistently cropped images for training. This can lead to suboptimal results for cropped images. If you experience quality issues with the `eynollah layout`, try setting the `-ipe/--ignore_page_extraction` option to skip the builtin cropping. We will rectify this in the next trainings.

## [0.7.0] - 2026-01-30

Added:

  * "Model zoo", central place to describe and load models, #207
  * Training code for the CNN/RNN OCR model

Changed:

  * Lint training code, #204
  * Update documentation: README, pyproject.toml metadata, guides in `docs/`, #209


## [0.6.0] - 2025-10-17

Added:

  * `eynollah-training` CLI and docs for training the models, #187, #193, https://github.com/qurator-spk/sbb_pixelwise_segmentation/tree/unifying-training-models

Fixed:

  * `join_polygons` always returning Polygon, not MultiPolygon, #203

## [0.6.0rc2] - 2025-10-14

Fixed:

  * Prevent OOM GPU error by avoiding loading the `region_fl` model, #199
  * XML output: encoding should be `utf-8`, not `utf8`, #196, #197

## [0.6.0rc1] - 2025-10-10

Fixed:

 * continue processing when no columns detected but text regions exist
 * convert marginalia to main text if no main text is present
 * reset deskewing angle to 0° when text covers <30% image area and detected angle >45°
 * :fire: polygons: avoid invalid paths (use `Polygon.buffer()` instead of dilation etc.)
 * `return_boxes_of_images_by_order_of_reading_new`: avoid Numpy.dtype mismatch, simplify
 * `return_boxes_of_images_by_order_of_reading_new`: log any exceptions instead of ignoring
 * `filter_contours_without_textline_inside`: avoid removing from duplicate lists twice
 * `get_marginals`: exit early if no peaks found to avoid spurious overlap mask
 * `get_smallest_skew`: after shifting search range of rotation angle, use overall best result
 * Dockerfile: fix CUDA installation (cuDNN contested between Torch and TF due to extra OCR)
 * OCR: re-instate missing methods and fix `utils_ocr` function calls
 * mbreorder/enhancement CLIs: missing imports
 * :fire: writer: `SeparatorRegion` needs `SeparatorRegionType` (not `ImageRegionType`), f458e3e
 * tests: switch from `pytest-subtests` to `parametrize` so we can use `pytest-isolate`
   (so CUDA memory gets freed between tests if running on GPU)

Added:
 * :fire: `layout` CLI: new option `--model_version` to override default choices
 * test coverage for OCR options in `layout`
 * test coverage for table detection in `layout`
 * CI linting with ruff

Changed:
 
 * polygons: slightly widen for regions and lines, increase for separators
 * various refactorings, some code style and identifier improvements
 * deskewing/multiprocessing: switch back to ProcessPoolExecutor (faster), 
   but use shared memory if necessary, and switch back from `loky` to stdlib,
   and shutdown in `del()` instead of `atexit`
 * :fire: OCR: switch CNN-RNN model to `20250930` version compatible with TF 2.12 on CPU, too
 * OCR: allow running `-tr` without `-fl`, too
 * :fire: writer: use `@type='heading'` instead of `'header'` for headings
 * :fire: performance gains via refactoring (simplification, less copy-code, vectorization,
   avoiding unused calculations, avoiding unnecessary 3-channel image operations)
 * :fire: heuristic reading order detection: many improvements
    - contour vs splitter box matching: 
      * contour must be contained in box exactly instead of heuristics
      * make fallback center matching, center must be contained in box
    - original vs deskewed contour matching:
      * same min-area filter on both sides
      * similar area score in addition to center proximity
      * avoid duplicate and missing mappings by allowing N:M
        matches and splitting+joining where necessary
 * CI: update+improve model caching


## [0.5.0] - 2025-09-26

Fixed:

  * restoring the contour in the original image caused an error due to an empty tuple, #154
  * removed NumPy warnings calculating sigma, mean,  (fixed issue #158)
  * fixed bug in `separate_lines.py`, #124
  * Drop capitals are now handled separately from their corresponding textline
  * Marginals are now divided into left and right. Their reading order is written first for left marginals, then for right marginals, and within each side from top to bottom
  * Added a new page extraction model. Instead of bounding boxes, it outputs page contours in the XML file, improving results for skewed pages
  * Improved reading order for cases where a textline is segmented into multiple smaller textlines

Changed

  * CLIs: read only allowed filename suffixes (image or XML) with `--dir_in`
  * CLIs: make all output option required, and `-i` / `-di` required but mutually exclusive
  * ocr CLI: drop redundant `-brb` in favour of just `-dib`
  * APIs: move all input/output path options from class (kwarg and attribute) ro `run` kwarg
  * layout textlines: polygonal also without `-cl`

Added:

  * `eynollah machine-based-reading-order` CLI to run reading order detection, #175
  * `eynollah enhancement` CLI to run image enhancement, #175
  * Improved models for page extraction and reading order detection, #175
  * For the lightweight version (layout and textline detection), thresholds are now assigned to the artificial class. Users can apply these thresholds to improve detection of isolated textlines and regions. To counteract the drawback of thresholding, the skeleton of the artificial class is used to keep lines as thin as possible (resolved issues #163 and #161)
  * Added and integrated a trained CNN-RNN OCR models
  * Added and integrated a trained TrOCR model
  * Improved OCR detection to support vertical and curved textlines
  * Introduced a new machine-based reading order model with rotation augmentation
  * Optimized reading order speed by clustering text regions that belong to the same block, maintaining top-to-bottom order
  * Implemented text merging across textlines based on hyphenation when a line ends with a hyphen
  * Integrated image enhancement as a separate use case
  * Added reading order functionality on the layout level as a separate use case
  * CNN-RNN OCR models provide confidence scores for predictions
  * Added OCR visualization: predicted OCR can be overlaid on an image of the same size as the input
  * Introduced a threshold value for CNN-RNN OCR models, allowing users to filter out low-confidence textline predictions
  * For OCR, users can specify a single model by name instead of always using the default model
  * Under the OCR use case, if Ground Truth XMLs and images are available, textline image and corresponding text extraction can now be performed

Merged PRs:

  * better machine based reading order + layout and textline + ocr by @vahidrezanezhad in https://github.com/qurator-spk/eynollah/pull/175
  * CI: pypi by @kba in https://github.com/qurator-spk/eynollah/pull/154
  * CI: Use most recent actions/setup-python@v5 by @kba in https://github.com/qurator-spk/eynollah/pull/157
  * update docker by @bertsky in https://github.com/qurator-spk/eynollah/pull/159
  * Ocrd fixes by @kba in https://github.com/qurator-spk/eynollah/pull/167
  * Updating readme for eynollah use cases cli by @kba in https://github.com/qurator-spk/eynollah/pull/166
  * OCR-D processor: expose reading_order_machine_based by @bertsky in https://github.com/qurator-spk/eynollah/pull/171
  * prepare release v0.5.0: fix logging by @bertsky in https://github.com/qurator-spk/eynollah/pull/180
  * mb_ro_on_layout: remove copy-pasta code not actually used by @kba in https://github.com/qurator-spk/eynollah/pull/181
  * prepare release v0.5.0: improve CLI docstring, refactor I/O path options from class to run kwargs, increase test coverage @bertsky in #182
  * prepare release v0.5.0: fix for OCR doit subtest by @bertsky in https://github.com/qurator-spk/eynollah/pull/183
  * Prepare release v0.5.0 by @kba in https://github.com/qurator-spk/eynollah/pull/178
  * updating eynollah README, how to use it for use cases by @vahidrezanezhad in https://github.com/qurator-spk/eynollah/pull/156
  * add feedback to command line interface by @michalbubula in https://github.com/qurator-spk/eynollah/pull/170

## [0.4.0] - 2025-04-07

Fixed:

 * allow empty imports for optional dependencies
 * avoid Numpy warnings (empty slices etc.)
 * remove deprecated Numpy types
 * binarization CLI: make `dir_in` usable again

Added:

 * Continuous Deployment via Dockerhub and GHCR
 * CI: also test CLIs and OCR-D
 * CI: measure code coverage, annotate+upload reports
 * smoke-test: also check results
 * smoke-test: also test sbb-binarize
 * ocrd-test: analog for OCR-D CLI (segment and binarize)
 * pytest: add asserts, extend coverage, use subtests for various options
 * pytest: also add binarization
 * pytest: add `dir_in` mode (segment and binarize)
 * make install: control optional dependencies via `EXTRAS` variable
 * OCR-D: expose and describe recently added parameters:
    - `ignore_page_extraction`
    - `allow_enhancement`
    - `textline_light`
    - `right_to_left`
 * OCR-D: :fire: integrate ocrd-eynollah-binarize
 * add detection confidence in `TextRegion/Coords/@conf`
   (but only in light version and not for marginalia)

Changed:

 * Docker build: simplify, w/ `OCR`, conform to OCR-D spec
 * OCR-D: :fire: migrate to core v3
    - initialize+setup only once
    - restrict number of parallel page workers to 1
      (conflicts with existing multiprocessing; TF parts not mp-compatible)
    - do query maximally annotated page image
      (but filtering existing binarization/cropping/deskewing),
      rebase (as new `@imageFilename`) if necessary
    - add behavioural docstring

 * :fire: refactor `Eynollah` API:
    - no more data (kw)args at init,
       but kwargs `dir_in` / `image_filename` for `run()`
    - no more data attributes, but function kwargs
      (`pcgts`, `image_filename`, `image_pil`, `dir_in`, `override_dpi`)
    - remove redundant TF session/model loaders
      (only load once during init)
    - factor `run_single()` out of `run()` (loop body),
      expose for independent calls (like OCR-D)
    - expose `cache_images()`, add `dpi` kwarg, set `self._imgs`
    - single-image mode writes PAGE file result
      (just as directory mode does)

 * CLI: assertions (instead of print+exit) for options checks
 * light mode: fine-tune ratio to better detect a region as header

## [0.3.1] - 2024-08-27

Fixed:

  * regression in OCR-D processor, #106
  * Expected Ptrcv::UMat for argument 'contour', #110
  * Memory usage explosion with very narrow images (e.g. book spine), #67

## [0.3.0] - 2023-05-13

Changed:

  * Eynollah light integration, #86
  * use PEP420 style qurator namespace, #97
  * set_memory_growth to all GPU devices alike, #100

Fixed:

  * PAGE-XML coordinates can have self-intersections, #20
  * reading order representation (XML order vs index), #22
  * allow cropping separately, #26
  * Order of regions, #51
  * error while running inference, #75
  * Eynollah crashes while processing image, #77
  * ValueError: bad marshal data, #87
  * contour extraction: inhomogeneous shape, #92
  * Confusing model dir variables, #93
  * New release?, #96

## [0.2.0] - 2023-03-24

Changed:

  * Convert default model from HDFS to TF SavedModel, #91

Added:

  * parameter `tables` to toggle table detection, #91
  * default model described in ocrd-tool.json, #91

## [0.1.0] - 2023-03-22

Fixed:

  * Do not produce spurious `TextEquiv`, #68
  * Less spammy logging, #64, #65, #71

Changed:

  * Upgrade to tensorflow 2.4.0, #74
  * Improved README
  * CI: test for python 3.7+, #90

## [0.0.11] - 2022-02-02

Fixed:

  * `models` parameter should have `content-type`, #61, OCR-D/core#777

## [0.0.10] - 2021-09-27

Fixed:

  * call to `uild_pagexml_no_full_layout` for empty pages, #52

## [0.0.9] - 2021-08-16

Added:

  * Table detection, #48

Fixed:

  * Catch exception, #47

## [0.0.8] - 2021-07-27

Fixed:

  * `pc:PcGts/@pcGtsId` was not set, #49

## [0.0.7] - 2021-07-01

Fixed:

  * `slopes`/`slopes_h` retval/arguments mixed up, #45, #46

## [0.0.6] - 2021-06-22

Fixed:

  * Cast arguments to opencv2 to python native types, #43, #44, opencv/opencv#20186

## [0.0.5] - 2021-05-19

Changed:

  * Remove `allow_enhancement` parameter, #42

## [0.0.4] - 2021-05-18

  * fix contour bug, #40

## [0.0.3] - 2021-05-11

  * fix NaN bug, #38

## [0.0.2] - 2021-05-04

Fixed:

  * prevent negative coordinates for textlines in marginals
  * fix a bug in the contour logic, #38
  * the binarization model is added into the models and now binarization of input can be done at the first stage of eynollah's pipline. This option can be turned on by -ib (-input_binary) argument. This is suggested for very dark or bright documents

## [0.0.1] - 2021-04-22

Initial release

<!-- link-labels -->
[0.9.2]: ../../compare/v0.9.2...v0.9.1
[0.9.1]: ../../compare/v0.9.1...v0.9.0
[0.9.0]: ../../compare/v0.9.0...v0.8.0
[0.8.0]: ../../compare/v0.8.0...v0.7.0
[0.7.0]: ../../compare/v0.7.0...v0.6.0
[0.6.0]: ../../compare/v0.6.0...v0.6.0rc2
[0.6.0rc2]: ../../compare/v0.6.0rc2...v0.6.0rc1
[0.6.0rc1]: ../../compare/v0.6.0rc1...v0.5.0
[0.5.0]: ../../compare/v0.5.0...v0.4.0
[0.4.0]: ../../compare/v0.4.0...v0.3.1
[0.3.1]: ../../compare/v0.3.1...v0.3.0
[0.3.0]: ../../compare/v0.3.0...v0.2.0
[0.2.0]: ../../compare/v0.2.0...v0.1.0
[0.1.0]: ../../compare/v0.1.0...v0.0.11
[0.0.11]: ../../compare/v0.0.11...v0.0.10
[0.0.10]: ../../compare/v0.0.10...v0.0.9
[0.0.9]: ../../compare/v0.0.9...v0.0.8
[0.0.8]: ../../compare/v0.0.8...v0.0.7
[0.0.7]: ../../compare/v0.0.7...v0.0.6
[0.0.6]: ../../compare/v0.0.6...v0.0.5
[0.0.5]: ../../compare/v0.0.5...v0.0.4
[0.0.4]: ../../compare/v0.0.4...v0.0.3
[0.0.3]: ../../compare/v0.0.3...v0.0.2
[0.0.2]: ../../compare/v0.0.2...v0.0.1
[0.0.1]: ../../compare/HEAD...v0.0.1
