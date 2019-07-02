__author__ = "Lei Wang"
__date__ = "28-Aug-2018"
__update__ = "21-March-2019"
__license__ = "MIT"

from __future__ import print_function

import os
import sys
import argparse
from prompt_toolkit import prompt
import chalk
import traceback

NOT_STARTED = 0
FAILED = -1
SUCC = 1
DELIMITER = ' '

class CommandBase:
    
    EXIT = "exit"
    HELP = "help"

    def __init__(self):
        self.status = NOT_STARTED
        self.delimiter = DELIMITER
        self.name = "Command"
        self.prompt_prolog = ""
        self.params = {
            "latestCmd": None,
            "history": [] # for central audits
        }
        self.usage = ""

    def promptLoop(self, doc):
        while True:
            command = prompt(self.prompt_prolog)
            try:
                self.comprehend(command.split(self.delimiter), doc)
            except StopIteration:
                break
            except SystemExit:
                pass

    def comprehend(self, raw_args, doc):
        parser = argparse.ArgumentParser(description=doc,usage=self.usage)
        parser.add_argument('command', help="SubCommand to be executed.")
        args = parser.parse_args(raw_args[0:1])
        if not hasattr(self, args.command):
            if args.command != HELP:
                print("Unrecognized command!")

            parser.print_help()
            self.status = FAILED
        
        else:
            self.dispatch(args.command, raw_args[1:], doc)

    def execute(self, prepared_command):
        raise NotImplemented

    def dispatch(self, command, raw_args, doc):
        handler = getattr(self, command)
        try:
            if command == EXIT:
                handler()
            else:
                handler(raw_args, doc)
        except StopIteration as err:
            raise(err)
        except Exception as err:
            print(chalk.red(err))
            traceback.print_exc()

    def __str__(self):
        return "<%s>" % self.__class__.__name__
