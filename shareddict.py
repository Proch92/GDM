class SharedDict():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(SharedDict, cls).__new__(cls)
        return cls.instance

    def add(self, key, value):
        if not hasattr(self, key):
            setattr(self, key, [])
        eval("self." + key).append(value)

    def to_dict(self):
        return self.__dict__
