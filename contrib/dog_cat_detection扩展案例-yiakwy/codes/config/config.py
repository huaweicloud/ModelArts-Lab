'''
Created on 15 Jul, 2016

@author: wangyi
'''

import os 
# from config import cfg
import importlib
import global_settings

ENVIRON_CONFIG = "config"

try:
  basestring
except NameError:
  basestring = str

class ImproperlyConfigured(Exception): pass 

class Settings:
    
    def __init__(self, custom_settings=None):
        # update global settings 
        for setting in dir(global_settings):
            if setting.isupper() and not setting.startswith("__"):
                setattr(self, setting, getattr(global_settings, setting))
                
        if custom_settings is None:
            custom_settings = os.environ.get(ENVIRON_CONFIG)
        if custom_settings is not None and isinstance(custom_settings, basestring):
            try:
                custom_settings = importlib.import_module(custom_settings)
            except Exception as ex:
                raise ImproperlyConfigured("")
        
        self._setting_module = custom_settings    
        if custom_settings is not None:
            self._overriden_vals = set()
            for setting in dir(custom_settings):
                if setting.isupper():
                    val = getattr(custom_settings, setting)
                    # do some checking
                    
                    # overriden
                    setattr(self, setting, val)
                    self._overriden_vals.add(setting)
                    
    def __str__(self):
        ret = []
        ret.append("\nConfigurations:\n")
        for setting in dir(self):
            if setting.isupper() and not setting.startswith("__"):
                ret.append("{:30} {}\n".format(setting, getattr(self, setting)))
        ret.append("\n")
        return "".join(ret)
        
    def __repr__(self):
        return "<Setting Object: {}>".format(self._setting_module.__name__)
