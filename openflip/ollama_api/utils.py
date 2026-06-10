import time
import re

COLOR_END = '\033[0m'
COLOR_RED = '\033[91m'
COLOR_GREEN = '\033[92m'
COLOR_YELLOW = '\033[93m'
COLOR_BLUE = '\033[94m'

def remove_color_tags(raw_text:str):
    return re.sub(r'\033\[\d+(;\d+)*m', '', raw_text)

def red(text):
    return f"{COLOR_RED}{remove_color_tags(text)}{COLOR_END}"

def yellow(text):
    return f"{COLOR_YELLOW}{remove_color_tags(text)}{COLOR_END}"

def blue(text):
    return f"{COLOR_BLUE}{remove_color_tags(text)}{COLOR_END}"


def timestamp():
    ts = time.localtime()
    h, m, s = ts.tm_hour, ts.tm_min, ts.tm_sec
    h, m, s = f"{h:02}", f"{m:02}", f"{s:02}"
    ts_string = f"[{h}:{m}:{s}]"
    return ts_string


def log_message(text:str, include_timestamp:bool = True):
    try:
        if include_timestamp:
            text = f"{timestamp()} {text}"
        with open ("log.txt", "a") as file:
            file.write(f"{remove_color_tags(text)}\n")
    except:
        pass


prev_ts_string = ""
def print_ts(text:str="", include_timestamp:bool=True, force_timestamp:bool=False, end:str="\n", error:bool = False, log:bool=True):
    global prev_ts_string
    ts = timestamp()
    if not force_timestamp and ts == prev_ts_string:
        ts = "          "
    else:
        prev_ts_string = ts
    formatted_text = f"{f'{ts} ' if include_timestamp else ''}{text}"
    if log:
        log_message(formatted_text, False)
    if error is True:
        formatted_text = f"{red(formatted_text)}"
    print(formatted_text, end=end)


def scrub_input(input:str) -> str:
    if not input:
        return input
    new_input = input.strip()
    new_input = re.sub(r' {2,}', ' ', new_input)
    return new_input


def parse_input(input:str, sep:str = ' ') -> list:
    if not input:
        return []
    if isinstance(input, list):
        return input
    if not isinstance(input, str):
        return [input]
    input = input.strip().replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    input_list = [i.strip() for i in input.split(sep) if i.strip()]
    return input_list


"""def load_config(force_load:bool = False):
    '''global config
    if isinstance(config, dict) and not force_load:
        print_ts("Loading cached config")
        return config'''
    current_dir = os.path.dirname(__file__)
    filepath = os.path.join(current_dir, "config.json")
    try:
        with open(filepath, "r") as json_file:
            config = json.loads(json_file.read())
        return config
    except FileNotFoundError:
        print_ts("Missing config file.")
    except Exception as e:
        print_ts(f"Unknown error while loading config: \"{e}\"")
    config = {}
    return config


def load_api_config(force_load:bool = False):
    '''global api_config
    if isinstance(api_config, dict) and not force_load:
        print_ts("Loading cached api config")
        return config'''
    current_dir = os.path.dirname(__file__)
    filepath = os.path.join(current_dir, "api_config.json")
    try:
        with open(filepath, "r") as json_file:
            api_config = json.load(json_file)
        return api_config
    except FileNotFoundError:
        print_ts("Missing 'api_config.json' file. Creating empty file now.")
        with open(filepath, "w") as json_file:
            json_file.write('{"discord_bot_token" : ""}')
    except Exception as e:
        print_ts(f"Unknown error while loading api config: \"{e}\"")
    api_config = {}
    return api_config"""