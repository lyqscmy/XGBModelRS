#!/usr/bin/env python3

import sys

model = "XGBModel"
level = "DEBUG"
timestamp = "2018-05-12T03:07:30Z"
log_format = "{} {}: {}: ".format(level, timestamp, model)

raw_log = sys.argv[1]
parsed_log = sys.argv[2]
with open(raw_log) as f1:
    with open(parsed_log, 'w') as f2:
        for line in f1:
            line = line[len(log_format):]
            f2.write(line)
