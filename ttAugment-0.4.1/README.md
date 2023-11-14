# TTAugment
![GitHub](https://img.shields.io/github/license/cypherics/TTAugment)
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![Downloads](https://pepy.tech/badge/ttaugment)

Perform Augmentation during Inference and aggregate the results of all the applied augmentation to create a
final output

## Installation

    pip install ttAugment


## Supported Augmentation
Library supports all [color](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_color.html), 
[blur](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_blur.html) and [contrast](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_contrast.html)
transformation provided by [imgaug](https://imgaug.readthedocs.io/en/latest/) along with custom Geometric Transformation.

1. Mirror : Crop an image to `crop_to_dimension` and mirror pixels to match the size of `window_dimension`
2. CropScale : Crop an image to `crop_to_dimension` and rescale the image to match the size of `window_dimension`
3. NoAugment : Keep the input unchanged
4. Crop : Crop an image to `crop_to_dimension`
5. Rot : Rotate an Image
6. FlipHorizontal
7. FlipVertical 

## Usage

How to use when test image is much **larger** than what the model requires, Don't worry the library has it covered,
it will generate fragments according to the specified dimension, so the inference can be performed while applying augmentation.

- window_size: Break the image into smaller images of said size 
- output_dimension: It must be greater the input image in order for the fragments to be restored back on the 
image.

```python
import numpy as np
from tt_augment.augment import generate_seg_augmenters

transformation_to_apply = [
  {"name": "Mirror", "crop_to_dimension": (256, 256)},
  {"name": "CropScale", "crop_to_dimension": (256, 256)},
]

for i in range(0, 10):
  image = np.random.rand(512, 512, 3) * 255
  image = np.expand_dims(image, 0)

  # Load augmentation object for the image, this includes to break the image in smaller fragments.
  tta = generate_seg_augmenters(
    image=image,
    window_size=(384, 384),
    output_dimension=(1, 512, 512, 3),
    transformation_to_apply=transformation_to_apply,
  )

  # Iterate over transformation_to_apply
  for iterator, transformation in enumerate(tta):
    # Iterate over individual fragments
    for augmented_fragment in transformation.transform_fragment():
      #     ---> transformed_fragment.shape = (1, 384, 384, 3) 
      # Inference steps for augmented fragment
      # 1. perform image normalization
      #     ---> normalised_image = image_normalization(augmented_fragment)
      # 2. perform model prediction
      #     ---> prediction = model.predict(normalised_image)
      # 3. convert prediction to numpy with shape [batch, h, w, channel]
      # 4. place the prediction fragment on its position in the original image
      #     ---> transformation.restore_fragment(prediction)

      transformation.restore_fragment(augmented_fragment)

  # Aggregate the result for the input image over all applied augmentations
  tta.merge()

  output = tta.tta_output()
```


    

