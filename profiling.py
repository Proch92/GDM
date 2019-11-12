

class Profiler():
    def __init__(self):
        self.instances_seen = set()
        self.categories_seen = set()
        self.sessions_seen = set()
        self.profiling = {
            'episode': [],
            'no_instances_seen': [],
            'no_categories_seen': [],
            'episodic': {
                'no_neurons': [],
                'aqe': [],
                'test_accuracy_inst': [],
                'test_accuracy_cat': [],
                'first_accuracy_inst': [],
                'first_accuracy_cat': [],
                'whole_accuracy_inst': [],
                'whole_accuracy_cat': []
            },
            'semantic': {
                'no_neurons': [],
                'aqe': [],
                'test_accuracy_inst': [],
                'test_accuracy_cat': [],
                'first_accuracy_inst': [],
                'first_accuracy_cat': [],
                'whole_accuracy_inst': [],
                'whole_accuracy_cat': []
            }
        }
