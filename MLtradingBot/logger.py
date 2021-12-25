from datetime import datetime
import glob
import logging
from logging import Formatter, handlers, StreamHandler, getLogger, DEBUG, INFO, WARN
import inspect
import os
from datetime import datetime

import coloredlogs


timestamp = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
log_file_name = f'mltrading_{timestamp}.log'  # logファイルの名前

class Logger:
    def __init__(
            self, 
            *, 
            log_level: logging = DEBUG, 
            log_stdout: bool = True, 
            save_log_dir: str = 'log'):
        self.log_folder_path = os.path.join(os.path.dirname(__file__), save_log_dir)  # logフォルダのパス
        self.log_backupcount = 2
        os.makedirs(self.log_folder_path, exist_ok=True)  # logフォルダが無ければ作成
        log_file_path = os.path.join(self.log_folder_path, log_file_name)
        coloredlogs.CAN_USE_BOLD_FONT = True
        coloredlogs.DEFAULT_FIELD_STYLES = {
            'asctime': {'color': 'green'},
            'hostname': {'color': 'magenta'},
            'levelname': {'color': 'blue', 'bold': True},
            'name': {'color': 'blue'},
            'programname': {'color': 'cyan'}}
        coloredlogs.DEFAULT_LEVEL_STYLES = {
            'critical': {'color': 'red', 'bold': True},
            'error': {'color': 'red'},
            'warning': {'color': 'yellow'},
            'notice': {'color': 'magenta'},
            'info': {'color': 'green'},
            'debug': {'color': 'green'},
            'spam': {'color': 'green', 'faint': True},
            'success': {'color': 'green', 'bold': True},
            'verbose': {'color': 'blue'}}
        fsplits = inspect.stack()[1].filename.split('/')
        name = fsplits[len(fsplits)-1]
        # ロガー生成
        self.logger = getLogger(name)
        self.logger.setLevel(log_level)
        formatter = Formatter(
            fmt="%(asctime)s.%(msecs)03d %(levelname)7s %(message)s [%(name)s]",
            datefmt="%Y/%m/%d %H:%M:%S")
        # サイズローテーション
        handler = handlers.RotatingFileHandler(
            filename=log_file_path,
            encoding='UTF-8',
            maxBytes=16777216,  # 2**24 (16MB)
            backupCount=self.log_backupcount)
        # ログファイル設定
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        # 標準出力用 設定
        if log_stdout:
            coloredlogs.install(level=log_level, logger=self.logger, fmt=formatter._fmt, datefmt=formatter.datefmt)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warn(self, msg):
        self.logger.warning(msg)

    def error(self, msg, *, exc_info=True):
        self.logger.error(msg, exc_info=exc_info)

    def critical(self, msg):
        self.logger.critical(msg)

    def remove_oldlog(self, *, max_log_num=30):
        """
        古いlogファイルを消去
        Parameters
        ----------
        max_log_num : int, optional
            logファイルの最大件数, by default 10
        """
        logfile_name = os.path.join(self.log_folder_path, '*.log')
        logs = glob.glob(logfile_name)
        if len(logs) > max_log_num:
            log_list = [[log, datetime.strptime(log[-18:-4], '%Y%m%d%H%M%S')] for log in logs]
            log_list = sorted(log_list, key=lambda s: s[1])
            remove_log_path = log_list[0][0]
            self.info(f'remove {remove_log_path}')
            os.remove(remove_log_path)

            for i in range(1, self.log_backupcount+1):
                remove_rotating_log_path = remove_log_path + f'.{i}'
                if os.path.exists(remove_rotating_log_path):
                    os.remove(remove_rotating_log_path)