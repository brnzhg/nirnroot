#  python -m bootstrap arg1 arg2

import logging.handlers
import os 
import sys
from pathlib import Path
import logging
import configparser
from configparser import ConfigParser
import click

from typing import Tuple, Optional, Iterator, Generator

APP_NAME = 'nirnroot'
NIRN_ENV_VAR = 'NIRNDIR'
CONFIG_FILENAME = 'nirn.ini'

DATA_DIRNAME = 'data'
    
class NirnEnv:

    def __init__(self, app_name: str, config: ConfigParser, nirn_dir: Path):
        self.app_name: str = app_name
        self.config: ConfigParser = config
        self.nirn_dir: Path = nirn_dir
        self.data_dir: Path = nirn_dir / DATA_DIRNAME
        
    

def find_nirndir_and_config(config_filepath_override: Optional[Path]) -> str | Tuple[Path, Path]:

    def try_get_dir_config_file(dir_to_try: Path) -> Optional[Path]:
        if config_filepath_override:
            return config_filepath_override
        fpath: Path = dir_to_try / CONFIG_FILENAME
        return fpath if fpath.is_file() else None

    #config_filepath: Optional[Path] = config_filepath_override
    env_dirname: Optional[str] = os.getenv(NIRN_ENV_VAR)
    if env_dirname:
        env_dir: Path = Path(env_dirname)
        env_file = try_get_dir_config_file(env_dir)
        if env_file:
            return (env_dir, env_file)
        else:
            return f"Failed to find config {CONFIG_FILENAME} in env dir {NIRN_ENV_VAR}."

    cwd: Path = Path.cwd()
    cwd_config_file = try_get_dir_config_file(cwd)
    if cwd_config_file:
        return (cwd, cwd_config_file)

    source_path: Path = Path(__file__).parent.parent
    source_config_file  = try_get_dir_config_file(source_path)
    if source_config_file:
        return (source_path, source_config_file)

    return f"Failed to find config {CONFIG_FILENAME}."


@click.group()
@click.option('--config', '--c', type=click.Path(exists=True, dir_okay=False))
@click.pass_context
def cli(ctx: click.Context, config: click.Path):

    args_config_filepath: Optional[Path] = None if not config else Path(str(config))
    find_nirndir_and_config_r = find_nirndir_and_config(args_config_filepath)

    nirn_dir: Path
    config_filepath: Path 
    if isinstance(find_nirndir_and_config_r, str):
        ctx.fail(find_nirndir_and_config_r)
    else:
        nirn_dir, config_filepath = find_nirndir_and_config_r

    cp = configparser.ConfigParser()
    cp.read(config_filepath)

    # set context
    nirn_env: NirnEnv = NirnEnv(APP_NAME, cp, nirn_dir)
    ctx.obj = nirn_env

    # set logging
    logpath = nirn_env.data_dir / 'logs' / f'{APP_NAME}.log'
    print(logpath)
    fh = logging.handlers.TimedRotatingFileHandler(filename=str(logpath), when='midnight', interval=1, backupCount=10)
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        #$filemode='w',
                        handlers=[fh]) #filename='/tmp/myapp.log',
    logger = logging.getLogger()
    logger.setLevel(cp['Logging']['level']) # DEBUG

    logger.debug(f'NirnDir set to {nirn_dir}')

    
@cli.command(name='collect')
@click.pass_context
def collect_command(ctx: click.Context):
    """Collect your mouse click data.
    """
    nirn_env: NirnEnv = ctx.obj
    click.echo('called collect')
    click.echo(nirn_env.app_name)

    pass
# sys.argv
