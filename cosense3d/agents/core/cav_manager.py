

class CAVManager:
    def __init__(self, modules, max_num_cavs=10):
        self.modules = modules
        self.cavs = [CAV() for i in range(max_num_cavs)]

    def update_cav_info(self, cav_info):
        pass


class CAV:
    def __init__(self, id=None, is_ego=False):
        self.id = id
        self.mapped_id = -1
        self.is_ego = is_ego

    def reset(self, id, is_ego, mapped_id):
        self.id = id
        self.is_ego = is_ego
        self.mapped_id = mapped_id

    def forward(self, data):
        tasks = []
        return tasks

