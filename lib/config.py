from decouple import Config, RepositoryIni
import os

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
INI_FILE = os.path.join(root_dir, 'settings.ini')
ini_config = Config(RepositoryIni(INI_FILE))


CAR_DETECTION_MODEL_PATH = ini_config.get('CAR_DETECTION_MODEL_PATH', cast=str,
                                                 default=os.path.join(root_dir, 'trained_models',
                                                                      'car_detection.onnx'))

CAR_CLASSIFICATION_MODEL_PATH = ini_config.get('CAR_CLASSIFICATION_MODEL_PATH', cast=str,
                                                 default=os.path.join(root_dir, 'trained_models',
                                                                      'car_classification.pt'))

DATABASE_PATH = ini_config.get('DATABASE_PATH', cast=str,
                                        default=os.path.join(root_dir, 'trained_models',
                                                            'database.yaml'))