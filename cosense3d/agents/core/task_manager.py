from collections import OrderedDict

class TaskManager:
    def __init__(self):
        pass

    def summarize_forward_tasks(self, tasks):
        tasks['no_grad'] = self.reformat_tasks(tasks['no_grad'])
        tasks['with_grad'] = self.reformat_tasks(tasks['with_grad'])
        return tasks

    def summarize_loss_tasks(self, tasks):
        return self.reformat_tasks(tasks)

    def reformat_tasks(self, task_list):
        task_out = {}
        for task in task_list:
            cav_id, task_label, args = task
            order, task_name = task_label.split(':')
            order = int(order)
            if order not in task_out:
                task_out[order] = {}
            if task_name not in task_out[order]:
                task_out[order][task_name] = []
            task_out[order][task_name].append((cav_id, args))
        orders = sorted(task_out)
        ordered_task = OrderedDict()
        for i in orders:
            for k, v in task_out[i].items():
                ordered_task[k] = v
        return ordered_task

