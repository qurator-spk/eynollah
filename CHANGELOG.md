Change Log
==========

Versioned according to [Semantic Versioning](http://semver.org/).

## Unreleased

## [0.4.0] - 2025-04-07

Fixed:

 * allow empty imports for optional dependencies
 * avoid Numpy warnings (empty slices etc)
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
 * OCR-D: :fire: integrate ocrd-sbb-binarize
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

  * parmeter `tables` to toggle table detectino, #91
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
