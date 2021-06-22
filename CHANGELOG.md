Change Log
==========

Versioned according to [Semantic Versioning](http://semver.org/).

## Unreleased

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
[0.0.6]: ../../compare/v0.0.6...v0.0.5
[0.0.5]: ../../compare/v0.0.5...v0.0.4
[0.0.4]: ../../compare/v0.0.4...v0.0.3
[0.0.3]: ../../compare/v0.0.3...v0.0.2
[0.0.2]: ../../compare/v0.0.2...v0.0.1
[0.0.1]: ../../compare/HEAD...v0.0.1
