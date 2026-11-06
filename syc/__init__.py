import os
import sys
import shutil

__version__ = "0.2.3"

try:
    current_temp = os.environ.get ("TEMP", os.environ.get ("TMP", ""))
    
    if current_temp:
        drive = os.path.splitdrive (os.path.abspath (current_temp)) [0]
    
        if drive:
            usage = shutil.disk_usage (drive)
    
            if usage.free < 50 * 1024 * 1024:
                local_tmp = os.path.abspath (".tmp")
                os.makedirs (local_tmp, exist_ok = True)
                os.environ ["TEMP"] = local_tmp
                os.environ ["TMP"] = local_tmp

except Exception:
    pass

# fix windows console encoding
def utf8_stdout ():
    streams = [sys.stdout, sys.stderr]
    
    for stream in streams:
        try:
            stream.reconfigure (encoding = "utf-8", errors = "replace")
        except Exception:
            pass