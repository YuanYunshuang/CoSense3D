from collections import OrderedDict

class TaskManager:
    def __init__(self):
        pass

    def summarize_tasks(self, tasks):
        tasks_out = {0: {'no_grad': [], 'with_grad': []},
                     1: {'no_grad': [], 'with_grad': []},
                     2: {'loss': []}}
        no_grad0, no_grad1, _ = self.reformat_tasks(tasks['no_grad'])
        with_grad0, with_grad1, _ = self.reformat_tasks(tasks['with_grad'])
        tasks_out[0]['no_grad'] = no_grad0
        tasks_out[0]['with_grad'] = with_grad0
        tasks_out[1]['no_grad'] = no_grad1
        tasks_out[1]['with_grad'] = with_grad1
        tasks_out[2]['loss'] = self.reformat_tasks(tasks['loss'])[2]
        return tasks_out

    def summarize_loss_tasks(self, tasks):
        return self.reformat_tasks(tasks)

    def reformat_tasks(self, task_list):
        task_out = ({}, {}, {})  # two stages
        if len(task_list) == 0:
            return task_out
        for task in task_list:
            cav_id, task_label, args = task
            stage_order, task_name = task_label.split(':')
            stage = int(stage_order[0])
            order = int(stage_order[1:])
            task_name = task_name.strip()
            if order not in task_out[stage]:
                task_out[stage][order] = {}
            if task_name not in task_out[stage][order]:
                task_out[stage][order][task_name] = []
            task_out[stage][order][task_name].append((cav_id, args))

        task_out = [self.task_to_ordered_dict(tasks) for tasks in task_out]
        return task_out

    def task_to_ordered_dict(self, tasks):
        orders = sorted(tasks)
        ordered_task = OrderedDict()
        for i in orders:
            for k, v in tasks[i].items():
                ordered_task[k] = v
        return ordered_task


class SeqTaskManager:
    def __init__(self, seq_len=0):
        self.seq_len = seq_len

    def summarize_tasks(self, tasks_in):
        tasks_out = {}
        for x in ['no_grad', 'with_grad', 'loss']:
            batched_tasks = self.reformat_tasks(tasks_in[x])
            for stage, tasks in batched_tasks.items():
                if stage not in tasks_out:
                    tasks_out[stage] = {}
                tasks_out[stage][x] = tasks
        return tasks_out

    def summarize_loss_tasks(self, tasks):
        return self.reformat_tasks(tasks)

    def reformat_tasks(self, task_list):
        task_out = {} # two stages
        if len(task_list) == 0:
            return task_out
        for task in task_list:
            task_id, task_label, args = task
            stage_order, task_name = task_label.split(':')
            stage = int(stage_order[0])
            order = int(stage_order[1:])
            task_name = task_name.strip()

            if stage not in task_out:
                task_out[stage] = {}
            if order not in task_out[stage]:
                task_out[stage][order] = {}
            if task_name not in task_out[stage][order]:
                task_out[stage][order][task_name] = []
            task_out[stage][order][task_name].append((task_id, args))

        task_out = self.task_to_ordered_dict(task_out)
        return task_out

    def task_to_ordered_dict(self, task_dict):
        ordered_tasks = {}
        for stage, tasks in task_dict.items():
            orders = sorted(tasks)
            ordered_tasks[stage] = OrderedDict()
            for i in orders:
                for k, v in tasks[i].items():
                    ordered_tasks[stage][k] = v
        return ordered_tasks

    def parallel_to_sequential(self, task_dict):
        task_dict_out = {}
        for k, tasks in task_dict.items():
            task_dict_out[k] = {i: OrderedDict() for i in range(self.seq_len)}
            for task, args in tasks.items():
                for arg in args:
                    seq_idx = int(arg[0].split('.')[-1])
                    if task not in task_dict_out[k][seq_idx]:
                        task_dict_out[k][seq_idx][task] = []
                    task_dict_out[k][seq_idx][task].append(arg)
        return task_dict_out


