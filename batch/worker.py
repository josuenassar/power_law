#!/usr/bin/env python
"""Worker for jobs
"""
from redis import Redis
from rq import Worker, Queue, Connection

import subprocess

def gen_arg_list(cmd, kwarg_dict=None):
    arg_list = cmd
    if kwarg_dict is not None:
        for k, v in kwarg_dict.items():
            arg_list.append('--{}'.format(k))
            arg_list.append(str(v))
    return arg_list


def subprocs_exec(cmds, kwargs):
    arg = gen_arg_list(cmds, kwargs)
    subprocess.call(arg, shell=False, cwd=kwargs['dir'])  # dev/null

connection = Redis()

if __name__ == '__main__':

    listen = ['gpu']

    with Connection(connection):
        worker = Worker(map(Queue, listen))
        worker.work()
