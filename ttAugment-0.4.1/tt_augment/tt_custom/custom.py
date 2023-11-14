from imgaug.augmenters import meta

from tt_augment.tt_custom.tt_fwd_bkd import (
    MirrorFWD,
    MirrorBKD,
    FlipLR,
    FlipUD,
    RotateFWD,
    RotateBKD,
    ScaleFWD,
    ScaleBKD,
    NoSegAug,
)


class TTCustom:
    """
    Custom Augmenter are required for Geometric Transformation, When a geometric transformation is applied on the
    the test image the reverse has to be applied on the predicted image, to get the prediction as per the
    original non transformed image

    If an Image is rotated by 90, then the prediction is performed on the transformed image, but the prediction has to
    reversed to match the real image

    """

    def __init__(
        self,
        fwd=None,
        segmentation_reverse=None,
        classification_reverse=None,
        window_dimension=None,
        crop_to_dimension=None,
    ):

        assert len(window_dimension) == 4, (
            "Expected image to have shape (batch, height, height, [channels]), "
            "got shape %s." % (window_dimension,)
        )
        assert len(crop_to_dimension) == 4, (
            "Expected image to have shape (batch, height, height, [channels]), "
            "got shape %s." % (crop_to_dimension,)
        )

        if fwd is not None:
            assert isinstance(
                fwd, meta.Augmenter
            ), "Expected to have fwd of type [meta.Augmenter], " "but received %s." % (
                type(fwd),
            )
        if segmentation_reverse is not None:
            assert isinstance(
                segmentation_reverse, meta.Augmenter
            ), "Expected to have fwd of type [meta.Augmenter], " "but received %s." % (
                type(segmentation_reverse),
            )
        if window_dimension < crop_to_dimension:
            raise ValueError(
                "Inference Dimension Can't Be Less Than Transform Dimension"
            )
        self.crop_to_dimension = crop_to_dimension
        self.window_dimension = window_dimension

        self._fwd = fwd

        if segmentation_reverse is None:
            self._segmentation_reverse = NoSegAug(
                self.window_dimension, self.crop_to_dimension
            )
        else:
            self._segmentation_reverse = segmentation_reverse
        self._classification_reverse = classification_reverse

    @property
    def fwd(self) -> meta.Augmenter:
        """
        This function applies transformation on the images which are about to be inferred, generally referred as
        test images

        :return:
        """
        return self._fwd

    @property
    def segmentation_reverse(self) -> meta.Augmenter:
        """
        This function applies transformation on the predicted images, The reason being, the geometric transformation
        applied on the test images have to be restored/reversed to get the back the original transformation.

        :return:
        """
        return self._segmentation_reverse

    @property
    def classification_reverse(self):
        """
        Transform back the classification output

        :return:
        """
        return self._classification_reverse

    def get_parameters(self):
        return [self.window_dimension, self.crop_to_dimension]


class Mirror(TTCustom):
    """
    Crop an image to crop_to_dimension and mirror the left pixel to match the size of window_dimension
    """

    def __init__(self, crop_to_dimension: tuple, window_dimension: tuple):
        super().__init__(
            window_dimension=window_dimension,
            crop_to_dimension=crop_to_dimension,
        )
        self._fwd = MirrorFWD(self.window_dimension, self.crop_to_dimension)
        self._seg = MirrorBKD(self.window_dimension, self.crop_to_dimension)
        self._classification = None

    @property
    def fwd(self) -> meta.Augmenter:
        return self._fwd

    @property
    def segmentation_reverse(self) -> meta.Augmenter:
        return self._seg

    @property
    def classification_reverse(self):
        return self._classification


class CropScale(TTCustom):
    """
    Crop an image to crop_to_dimension and rescale the image to match the size of window_dimension
    """

    def __init__(self, crop_to_dimension: tuple, window_dimension: tuple):
        super().__init__(
            window_dimension=window_dimension,
            crop_to_dimension=crop_to_dimension,
        )
        if crop_to_dimension >= window_dimension:
            raise ValueError("Can't Scale array with same Dimension")
        self._fwd = ScaleFWD(self.window_dimension, self.crop_to_dimension)
        self._seg = ScaleBKD(self.window_dimension, self.crop_to_dimension)
        self._classification = None

    @property
    def fwd(self) -> meta.Augmenter:
        return self._fwd

    @property
    def segmentation_reverse(self) -> meta.Augmenter:
        return self._seg

    @property
    def classification_reverse(self):
        return self._classification


class NoAugment(TTCustom):
    def __init__(self, crop_to_dimension: tuple, window_dimension: tuple):
        assert window_dimension == crop_to_dimension, (
            "While Using NoAugment Transformation ",
            "Expected [window_dimension] and [crop_to_dimension] to be equal",
            "got %s and %s",
            (crop_to_dimension, window_dimension),
        )
        super().__init__(
            window_dimension=window_dimension,
            crop_to_dimension=crop_to_dimension,
        )
        if window_dimension != crop_to_dimension:
            raise ValueError("Dimension Mis Match")
        self._fwd = NoSegAug(self.window_dimension, self.crop_to_dimension)
        self._seg = NoSegAug(self.window_dimension, self.crop_to_dimension)
        self._classification = None

    @property
    def fwd(self) -> meta.Augmenter:
        return self._fwd

    @property
    def segmentation_reverse(self) -> meta.Augmenter:
        return self._seg

    @property
    def classification_reverse(self):
        return self._classification


class Crop(NoAugment):
    """
    Crop an image to crop_to_dimension
    """

    def __init__(self, crop_to_dimension: tuple, window_dimension: tuple):
        super().__init__(
            window_dimension=window_dimension,
            crop_to_dimension=crop_to_dimension,
        )


class Rot(TTCustom):
    """
    Rotate an image
    """

    def __init__(self, crop_to_dimension: tuple, window_dimension: tuple, angle: int):
        assert window_dimension == crop_to_dimension, (
            "While Using Geometric Transformation ",
            "Expected [window_dimension] and [crop_to_dimension] to be equal",
            "got %s and %s",
            (crop_to_dimension, window_dimension),
        )
        super().__init__(
            window_dimension=window_dimension,
            crop_to_dimension=crop_to_dimension,
        )

        angle_axis = [0, 90, 180, 270]
        assert angle in angle_axis, (
            "Expected angle to be [0,  90, 180, 270]",
            "given %s",
            (angle,),
        )

        self._fwd = RotateFWD(angle_axis.index(angle), self.crop_to_dimension)
        self._seg = RotateBKD(angle_axis.index(angle), self.crop_to_dimension)
        self._classification = None

    @property
    def fwd(self) -> meta.Augmenter:
        return self._fwd

    @property
    def segmentation_reverse(self) -> meta.Augmenter:
        return self._seg

    @property
    def classification_reverse(self):
        return self._classification


class FlipHorizontal(TTCustom):
    """
    Flip an image
    """

    def __init__(self, crop_to_dimension: tuple, window_dimension: tuple):
        assert window_dimension == crop_to_dimension, (
            "While Using Geometric Transformation ",
            "Expected [window_dimension] and [crop_to_dimension] to be equal",
            "got %s and %s",
            (crop_to_dimension, window_dimension),
        )
        super().__init__(
            window_dimension=window_dimension,
            crop_to_dimension=crop_to_dimension,
        )

        self._fwd = FlipLR(self.crop_to_dimension)
        self._seg = FlipLR(self.crop_to_dimension)
        self._classification = None

    @property
    def fwd(self) -> meta.Augmenter:
        return self._fwd

    @property
    def segmentation_reverse(self) -> meta.Augmenter:
        return self._seg

    @property
    def classification_reverse(self):
        return self._classification


class FlipVertical(TTCustom):
    """
    Flip an image
    """

    def __init__(self, crop_to_dimension: tuple, window_dimension: tuple):
        assert window_dimension == crop_to_dimension, (
            "While Using Geometric Transformation ",
            "Expected [window_dimension] and [crop_to_dimension] to be equal",
            "got %s and %s",
            (crop_to_dimension, window_dimension),
        )
        super().__init__(
            window_dimension=window_dimension,
            crop_to_dimension=crop_to_dimension,
        )
        self._fwd = FlipUD(self.crop_to_dimension)
        self._seg = FlipUD(self.crop_to_dimension)
        self._classification = None

    @property
    def fwd(self) -> meta.Augmenter:
        return self._fwd

    @property
    def segmentation_reverse(self) -> meta.Augmenter:
        return self._seg

    @property
    def classification_reverse(self):
        return self._classification
