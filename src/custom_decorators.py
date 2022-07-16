"""

Custom decorators for use with Metaflow

"""
from functools import wraps


def pip(libraries):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            import os
            import subprocess
            import sys
            IS_AWS_BATCH = os.getenv("AWS_BATCH_JOB_ID", None)
            if IS_AWS_BATCH:
                for library, version in libraries.items():
                    print("Pip Install:", library, version)
                    if version != "":
                        subprocess.run([sys.executable, "-m", "pip", "install", library + "==" + version])
                    else:
                        subprocess.run([sys.executable, "-m", "pip", "install", library])
            return function(*args, **kwargs)
        return wrapper
    return decorator


def enable_decorator(dec, flag):
    try:
        flag = bool(int(flag))
    except Exception as e:
        flag = False
        print(e)
        print("Assuming decorator is False for flag {}".format(flag))

    def decorator(func):
        if flag:
            return dec(func)
        return func
    return decorator
