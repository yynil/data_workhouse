from sqlitedict import SqliteDict
class SqliteDictWrapper:

    def __init__(self,file_name) -> None:
        self.file_name = file_name
        self.db = SqliteDict(file_name)

    def __getitem__(self, key):
        return self.db[key]
    
    def __setitem__(self, key, value):
        self.db[key] = value

    def __delitem__(self, key):
        del self.db[key]

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.close()

    def batch_update(self,update_dict):
        for key in update_dict:
            self.db[key] = update_dict[key]
        self.db.commit()

    def __len__(self):
        return len(self.db)
    
    def delall(self):
        for key in self.db:
            del self.db[key]
        self.db.commit()

    def __iter__(self):
        return self.db.__iter__()

if __name__ == '__main__':
    import random
    import uuid
    wrapper = SqliteDictWrapper('test.db')
    wrapper.delall()
    for i in range(100):
        id = str(uuid.uuid4())
        embedding = [random.random() for i in range(1024)]
        doc = ' '.join([str(random.randint(0,100)) for x in range(400)])
        wrapper[id] = {'embedding':embedding,'doc':doc}
    wrapper.commit()

    for key in wrapper.db:
        print(key,wrapper.db[key])

    print(f'length of db is {len(wrapper)}')
    wrapper.close()