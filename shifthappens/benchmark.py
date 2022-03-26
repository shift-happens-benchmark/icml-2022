

def _collect_tasks():
    return []

def _is_compatible(tasks, model):
    # TODO
    # based on defined Mixins, check if model and task
    # are compatible
    return True

class ScoreCard():

    def summary():
        """return dataframe with
        
        index         |   columns
        --------------------------
        task
        """
        pass

    def per_task(self, task):
        pass



def run(model):
    tasks = _collect_tasks()
    results = ScoreCardEntry()

    for task_cls in tasks:
        if not _is_compatible(task, model):
            continue
        for task in task_cls.iterate_instances():
            results.add(task.run())

        
