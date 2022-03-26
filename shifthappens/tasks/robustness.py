
__all__ = ["..."]

class ImageNetDataset(Dataset):

    def __init__(self):
        pass

    def __iter__(self):
        pass

class ImageNetRBase(Task):
    ...
    
    def setup():
        # pull data
        self._data = np.zeros(50000,...)
        self._labels = np.zeros(50000,...)
        self._dataset = TensorDataset(...)

        ###


class ImageNetRAdHoc(ImageNetRBase):

    def _evaluate():
        dataloader = Dataloader(..., max_batchsize = 1)
        scores = model.eval(dataloader)
        acc = (scores.predicted_classes == self._labels).mean()
        return Result(
            accuracy = acc
        )

@task(name = ..., standalone = True, other_global_attributes = False)
class ImageNetRBatchSizeAdapted(ImageNetRBase, CalibrationMixin):
    """ 
    index                          columns
    name    adapt_batch_size    |  accuracy , ... , ...

    Summary metrics --> percentile (?)
    task       metric     | value
    ImageNet-R error      | 50%
    ImageNet-R calbration | 20%
    """

    adapt_batch_size: int = parameter(default = 5, options=(2, 5, 9), doc = "the batch size")
    
    def _evaluate():
        dataloader = Dataloader(self.dataset, max_batchsize = self.adapt_batch_size) # for performance reasons
        scores = model.eval(dataloader)
        acc = (scores.predicted_classes == self._labels).mean()
        return Result(
            accuracy = acc,
            brier = brier_score
            ...
            summary_metric = {
                metrics.robustness : ("error", "mCE"),
                metrics.calibration : "calibration_score",
                ...
            }
        )


class ImageNetRAdapted(ImageNetRBase):
    
    def _prepare():
        dataloader = Dataloader(self.dataset, max_batchsize = None)
        self.model.prepare(dataloader)

    def _evaluate():
        dataloader = Dataloader(self.dataset, max_batchsize = None) # for performance reasons
        scores = model.eval(dataloader)
        acc = (scores.predicted_classes == self._labels).mean()
        return Result(
            accuracy = acc
        )

class TaskCollection():

    def _evaluate():
        # run all subtasks
        results = ...
        # do sth
        return Result(
            ...
        )
