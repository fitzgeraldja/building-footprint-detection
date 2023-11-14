from typing import List

from tt_augment import tt_custom

from image_fragment.fragment import ImageFragment

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import numpy as np

from tt_augment.tt_custom import custom, TTCustom


class TransformationOnImage:
    def __init__(
        self, name: str, restore_to: tuple, input_image: np.ndarray, collection: List
    ):
        self.name = name
        self.collection = collection
        self._input_image = input_image

        self._ongoing_transformation_window = None
        self.restore_to = np.zeros(restore_to)

    def transform_fragment(self):
        raise NotImplementedError

    def restore_fragment(self, image: np.ndarray):
        raise NotImplementedError


class SegmentationTransformationOnImage(TransformationOnImage):
    def __init__(
        self, name: str, restore_to: tuple, input_image: np.ndarray, collection: List
    ):
        super().__init__(name, restore_to, input_image, collection)

    def transform_fragment(self):
        for transformer_iterator, self._ongoing_transformation_window in enumerate(
            self.collection
        ):
            yield self._ongoing_transformation_window.transformer.fwd(
                images=self._ongoing_transformation_window.get_windowed_image(
                    image=self._input_image
                )
            )

    def restore_fragment(self, image: np.ndarray):
        restored_image = (
            self._ongoing_transformation_window.transformer.segmentation_reverse(
                images=image
            )
        )

        self.restore_to = (
            self._ongoing_transformation_window.fragment.transfer_fragment(
                transfer_from=restored_image, transfer_to=self.restore_to
            )
        )


class TransformationsPerRun(list):
    def __init__(self, merge_transformations="mean", transformation_on_image=None):
        if transformation_on_image is None:
            list.__init__(self, [])
        elif isinstance(transformation_on_image, TransformationOnImage):
            list.__init__(self, [transformation_on_image])
        elif isinstance(transformation_on_image, Iterable):
            assert all(
                [
                    isinstance(per_image, TransformationOnImage)
                    for per_image in transformation_on_image
                ]
            ), (
                "Expected all children to be FragmentTransformationPerImage, got types %s."
                % (", ".join([str(type(v)) for v in transformation_on_image]))
            )
            list.__init__(self, transformation_on_image)
        else:
            raise Exception(
                "Expected None or FragmentTransformationPerImage or list of"
                " FragmentTransformationPerImage, "
                "got %s." % (type(transformation_on_image),)
            )

        assert merge_transformations in ["mean"], (
            "Expected merge_transformations to be in ['mean']",
            "got %s",
            (merge_transformations,),
        )
        self.merge_transformations = merge_transformations
        self._output = None
        self._ongoing_transformation_on_image = None

    def add(self, transformation_on_image: TransformationOnImage):
        self.append(transformation_on_image)

    def merge(self):
        for transformation_per_image in self:
            if self._output is None:
                self._output = transformation_per_image.restore_to

            else:
                if self.merge_transformations == "mean":
                    self._output = self._output + transformation_per_image.restore_to

    def aggregate(self):
        if self.merge_transformations == "mean":
            self._output = self._output / len(self)

    def tta_output(self):
        self.aggregate()
        return self._output


class Window:
    def __init__(self, transformer, fragment):

        self._transformer = transformer
        self._name = self.transformer.__class__.__name__
        self._fragment = fragment

        self._output = None

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = value

    @property
    def fragment(self):
        return self._fragment

    @property
    def name(self):
        return self._name

    @property
    def transformer(self):
        return self._transformer

    def get_windowed_image(self, image: np.ndarray) -> np.ndarray:
        return self.fragment.get_fragment_data(image)


def _generate_transformation_to_apply(
    image_dimension: tuple,
    window_dimension: tuple,
    output_dimension: tuple,
    transformation_to_apply: list,
):
    assert len(image_dimension) == 4, (
        "Expected image to have shape (batch, height, width, [channels]), "
        "got shape %s." % (image_dimension,)
    )

    assert len(window_dimension) == 4, (
        "Expected image to have shape (batch, height, width, [channels]), "
        "got shape %s." % (window_dimension,)
    )

    assert len(output_dimension) == 4, (
        "Expected image to have shape (batch, height, width, [channels]), "
        "got shape %s." % (output_dimension,)
    )
    b, h, w, c = image_dimension

    for individual_transformer in transformation_to_apply:
        transformer_name = individual_transformer["name"]

        if "crop_to_dimension" not in list(individual_transformer.keys()):
            crop_to_dimension = window_dimension
        else:
            crop_to_size = individual_transformer["crop_to_dimension"]
            crop_to_dimension = (b, crop_to_size[0], crop_to_size[1], c)

        if "param" not in list(individual_transformer.keys()):
            transformer_param = {}
        else:
            transformer_param = individual_transformer["param"]

        transformer = look_up(
            transformer_name,
            crop_to_dimension,
            window_dimension,
            **transformer_param,
        )
        if transformer.crop_to_dimension > image_dimension:
            raise ValueError(
                "Transformation Dimension Can't be bigger that Image Dimension"
            )
        fragments = ImageFragment.image_fragment_4d(
            fragment_size=transformer.crop_to_dimension, org_size=image_dimension
        )
        apply_per_fragment = list()
        for fragment in fragments:
            apply_per_fragment.append(
                Window(transformer=transformer, fragment=fragment)
            )

        yield transformer_name, apply_per_fragment


def generate_seg_augmenters(
    image: np.ndarray,
    window_size: tuple,
    output_dimension: tuple,
    transformation_to_apply: list,
):
    assert len(image.shape) == 4, (
        "Expected image to have shape (batch, height, width, [channels]), "
        "got shape %s." % (image.shape,)
    )

    b, h, w, channels = image.shape
    window_dimension = (b, window_size[0], window_size[1], channels)

    transformations = TransformationsPerRun()
    for transformer_name, transformer_fragments in _generate_transformation_to_apply(
        image_dimension=image.shape,
        window_dimension=window_dimension,
        output_dimension=output_dimension,
        transformation_to_apply=transformation_to_apply,
    ):
        transformations.add(
            transformation_on_image=SegmentationTransformationOnImage(
                name=transformer_name,
                collection=transformer_fragments,
                input_image=image,
                restore_to=output_dimension,
            )
        )

    return transformations


def look_up(transformer_name, crop_to_dimension, window_dimension, **transformer_param):
    if hasattr(custom, transformer_name):
        custom_aug = getattr(custom, transformer_name)(
            crop_to_dimension=crop_to_dimension,
            window_dimension=window_dimension,
            **transformer_param,
        )
        return custom_aug
    elif hasattr(tt_custom, transformer_name):
        assert window_dimension == crop_to_dimension, (
            "While Using External Color Augmentation ",
            "Expected [window_dimension] and [crop_to_dimension] to be equal",
            "got %s and %s",
            (crop_to_dimension, window_dimension),
        )
        custom_aug = getattr(tt_custom, transformer_name)(**transformer_param)
        return TTCustom(
            fwd=custom_aug,
            window_dimension=crop_to_dimension,
            crop_to_dimension=crop_to_dimension,
        )
    else:
        raise Exception("UnSupported Transformer %s." % (transformer_name,))
