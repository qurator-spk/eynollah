When using Eynollah in OCR-D, the source image file group with (preferably) RGB images should be used as input like this:

    ocrd-eynollah-segment -I OCR-D-IMG -O OCR-D-SEG -P models eynollah_layout_v0_5_0

If the input file group is PAGE-XML (from a previous OCR-D workflow step), Eynollah behaves as follows:
- existing regions are kept and ignored (i.e. in effect they might overlap segments from Eynollah results)
- existing annotation (and respective `AlternativeImage`s) are partially _ignored_:
  - previous page frame detection (`cropped` images)
  - previous derotation (`deskewed` images)
  - previous thresholding (`binarized` images)
- if the page-level image nevertheless deviates from the original (`@imageFilename`)
  (because some other preprocessing step was in effect like `denoised`), then
  the output PAGE-XML will be based on that as new top-level (`@imageFilename`)

      ocrd-eynollah-segment -I OCR-D-XYZ -O OCR-D-SEG -P models eynollah_layout_v0_5_0

In general, it makes more sense to add other workflow steps **after** Eynollah.

There is also an OCR-D processor for binarization:

    ocrd-sbb-binarize -I OCR-D-IMG -O OCR-D-BIN -P models default-2021-03-09
