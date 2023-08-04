import os
from decouple import Config, RepositoryIni

root_dir = os.path.join(os.path.dirname(__file__), '..')
ini_file = os.path.join(root_dir, 'settings.ini')
ini_config = Config(RepositoryIni(ini_file))

MODEL_PATH = ini_config.get('MODEL_PATH', cast=str,
                    default=os.path.join(root_dir, 'trained_models', 'model.onnx'))
DATA_PATH = ini_config.get('DATA_PATH', cast=str,
                    default=os.path.join(root_dir, 'trained_models', 'data.pkl'))
LIMIT_LINE_PATH = ini_config.get('LIMIT_LINE_PATH', cast=str,
                    default=os.path.join(root_dir, 'trained_models', 'limit_line.json'))