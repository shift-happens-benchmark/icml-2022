"""Task mixins indicate the requirements of the task on the model in terms of the 
model's supported prediction types."""


class LabelTaskMixin:
    """Indicates that the task requires the model to return the predicted labels.

    Tasks implementing this mixin will be provided with the ``class_labels`` attribute in the
    :py:class:`shifthappens.models.base.ModelResult` returned during evaluation.
    """

    pass


class ConfidenceTaskMixin:
    """Indicates that the task requires the model to return the confidence scores.

    Tasks implementing this mixin will be provided with the ``confidences`` attribute in the
    :py:class:`shifthappens.models.base.ModelResult` returned during evaluation.
    """

    pass


class UncertaintyTaskMixin:
    """Indicates that the task requires the model to return the uncertainty scores.

    Tasks implementing this mixin will be provided with the ``uncertainties`` attribute in the
    :py:class:`shifthappens.models.base.ModelResult` returned during evaluation.
    """

    pass


class OODScoreTaskMixin:
    """Indicates that the task requires the model to return the OOD scores.

    Tasks implementing this mixin will be provided with the ``ood_scores`` attribute in the
    :py:class:`shifthappens.models.base.ModelResult` returned during evaluation.
    """

    pass


class FeaturesTaskMixin:
    """Indicates that the task requires the model to return the raw features.

    Tasks implementing this mixin will be provided with the ``features`` attribute in the
    :py:class:`shifthappens.models.base.ModelResult` returned during evaluation.
    """

    pass
