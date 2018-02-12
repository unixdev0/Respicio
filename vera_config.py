class Setup:
    path_img = 'image'
    path_raw = 'raw'
    path_inp = 'input'
    path_model = 'model'
    path_model_backup = 'backup'
    path_train = 'train'
    path_template = 'template'
    name_model = 'model.vera'
    name_vocab = 'vocab.vera'
    file_system_retries = 9
    file_system_retry_delay = 1
    model_config = 'models.json'
    model = 'tiff'
    image_ext = '.tiff'
    image_enhancement = True

class Config:
    vera = 'VERA_ROOT'
    debug = False
    wait_for_event_timeout = 5
    activation = 'sigmoid'
    threshold = 0.5