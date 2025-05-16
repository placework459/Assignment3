import json
import pickle
import os
from pathlib import Path

class Serilizer:

    def save_data(self, obj, file_path):
        with open(file_path,'wb') as f:
            pickle.dump(obj, f)
        print(f"object save to {file_path}")

    def load_from_pickle (self,file_path):
        with open(file_path, 'wb') as f:
            obj= pickle.load(f)
            print(f"loadfile {file_path}")
            return obj 