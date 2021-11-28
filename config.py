import os

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

DATA_IN = os.path.join(DATA_DIR, 'in')
DATA_OUT = os.path.join(DATA_DIR, 'out')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
CHPK_DIR = os.path.join(MODELS_DIR, 'checkpoints')
LOGS_DIR = os.path.join(MODELS_DIR, 'logs')

DOCS_DIR = os.path.join(ROOT_DIR, 'docs')
FIGS_DIR = os.path.join(DATA_DIR, 'figures')

UTILS_DIR = os.path.join(ROOT_DIR, 'utils')
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
