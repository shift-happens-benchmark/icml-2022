"""Model mixins indicate the supported prediction types of a model."""


class LabelModelMixin:
    """Inherit from this class if your model returns predicted labels."""

    pass


class ConfidenceModelMixin:
    """Inherit from this class if your model returns confidences."""

    pass


class UncertaintyModelMixin:
    """Inherit from this class if your model returns uncertainties."""

    pass


class OODScoreModelMixin:
    """Inherit from this class if your model returns ood scores."""

    pass


class FeaturesModelMixin:
    """Inherit from this class if your model returns features."""

    pass
