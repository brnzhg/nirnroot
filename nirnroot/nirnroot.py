#  python -m bootstrap arg1 arg2

import os 
import sys
from pathlib import Path
import logging
import configparser
import argparse

from typing import Tuple, Optional, Iterator, Generator

APP_NAME = 'Nirnroot'
NIRN_ENV_VAR = 'NIRNDIR'
CONFIG_FILENAME = 'nirn.conf'

class NirnInitException(BaseException):
    pass
    

def find_nirndir_and_config(config_filepath_override: Optional[Path]) -> Optional[Tuple[Path, Path]]:

    def try_get_dir_config_file(dir_to_try: Path) -> Optional[Path]:
        if config_filepath_override:
            return config_filepath_override
        fpath: Path = dir_to_try / CONFIG_FILENAME
        return fpath if fpath.is_file() else None

    #config_filepath: Optional[Path] = config_filepath_override
    env_dirname: str = os.getenv(NIRN_ENV_VAR)
    if env_dirname:
        env_dir: Path = Path(env_dirname)
        env_file = try_get_dir_config_file(env_dir)
        if env_file:
            return (env_dir, env_file)
        else:
            print(f"Failed to find config {CONFIG_FILENAME} in env dir {NIRN_ENV_VAR}.")
            return None

    cwd: Path = Path.cwd()
    cwd_config_file = try_get_dir_config_file(cwd)
    if cwd_config_file:
        return (cwd, cwd_config_file)

    source_path: Path = Path(__file__).parent.parent
    source_config_file  = try_get_dir_config_file(source_path)
    if source_config_file:
        return (source_path, source_config_file)
    
    return None

def main():

    parser = argparse.ArgumentParser(prog="Nirnroot")
    parser.add_argument("-c", "-config", type=str, dest='config_filename', help="config file")

    args = parser.parse_args()


    if args.config_filename:
        
    
    
    config = configparser.ConfigParser()
    config.read()






# sys.argv
