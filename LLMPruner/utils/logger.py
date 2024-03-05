import os
import sys
import time
import codecs
import logging


class LoggerWithDepth():
    def __init__(self, env_name, config, root_dir = 'runtime_log', overwrite = True, setup_sublogger = True):
        if os.path.exists(os.path.join(root_dir, env_name)) and not overwrite:
            raise Exception("Logging Directory {} Has Already Exists. Change to another name or set OVERWRITE to True".format(os.path.join(root_dir, env_name)))
        
        self.env_name = env_name
        self.root_dir = root_dir
        self.log_dir = os.path.join(root_dir, env_name)
        self.overwrite = overwrite

        # 定义输出log的格式
        self.format = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                        "%Y-%m-%d %H:%M:%S")

        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        
        # Save Hyperparameters
        self.write_description_to_folder(os.path.join(self.log_dir, 'description.txt'), config)
        self.best_checkpoint_path = os.path.join(self.log_dir, 'pytorch_model.bin')

        if setup_sublogger:
            sub_name = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()) # 时间作为sub_name
            self.setup_sublogger(sub_name, config)

    # 创建sublogger
    def setup_sublogger(self, sub_name, sub_config):
        self.sub_dir = os.path.join(self.log_dir, sub_name)
        # 检查目录是否存在，不存在则创建
        if os.path.exists(self.sub_dir):
            raise Exception("Logging Directory {} Has Already Exists. Change to another sub name or set OVERWRITE to True".format(self.sub_dir))
        else:
            os.mkdir(self.sub_dir)

        self.write_description_to_folder(os.path.join(self.sub_dir, 'description.txt'), sub_config)
        # 创建一个train.sh脚本，参数和运行参数一致
        with open(os.path.join(self.sub_dir, 'train.sh'), 'w') as f:
            f.write('python ' + ' '.join(sys.argv))

        # Setup File/Stream Writer
        # logging.Formatter 定义输出log的格式  
        # %(asctime)s、%(levelname)s 和 %(message)s 分别代表时间戳、日志级别和日志消息
        # 第二个参数定义了时间戳的格式，这里是年-月-日 时:分:秒
        log_format=logging.Formatter("%(asctime)s - %(levelname)s :       %(message)s", "%Y-%m-%d %H:%M:%S")
        # 通过 getLogger() 函数获取一个日志记录器。这里没有指定名称，因此返回的是根日志记录器
        self.writer = logging.getLogger()
        # 创建一个 FileHandler，它负责将日志消息写入磁盘文件
        fileHandler = logging.FileHandler(os.path.join(self.sub_dir, "training.log"))
        fileHandler.setFormatter(log_format)
        self.writer.addHandler(fileHandler)
        # 创建一个 consoleHandler，它负责将日志消息输出到控制台
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(log_format)
        # 将控制台处理器添加到日志记录器上    实现记录日志时会同时将日志输出到控制台
        self.writer.addHandler(consoleHandler)

        '''
        设置日志记录器的级别为 INFO。
        这意味着 INFO 级别及以上（如 WARNING、ERROR、CRITICAL）的日志消息会被处理和输出
        而低于 INFO 级别的日志消息（如 DEBUG）将被忽略
        '''
        self.writer.setLevel(logging.INFO)

        # Checkpoint
        self.checkpoint_path = os.path.join(self.sub_dir, 'pytorch_model.bin')      
        self.lastest_checkpoint_path = os.path.join(self.sub_dir, 'latest_model.bin')   

    def log(self, info):
        # info(self, msg, *args, **kwargs):
        # 日志"msg%args"，严重性为"INFO"。
        self.writer.info(info)

    def write_description_to_folder(self, file_name, config):
        with codecs.open(file_name, 'w') as desc_f:
            desc_f.write("- Training Parameters: \n")
            for key, value in config.items():
                desc_f.write("  - {}: {}\n".format(key, value))
