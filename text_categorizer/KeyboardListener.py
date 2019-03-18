#!/usr/bin/python3
# coding=utf-8

class KeyboardListener:

    __slots__ = ("listener")    # Based on http://www.qtrac.eu/pysavemem.html (accessed on 2019-03-15).

    def __init__(self, on_press=None, on_release=None):
        from pynput.keyboard import Listener
        self.listener = Listener(on_press=on_press, on_release=on_release) 

    def stop(self):
        self.listener.stop()

    @staticmethod
    def available():
        try:
            from pynput.keyboard import Listener
            return True
        except Exception:
            return False

    @staticmethod
    def how_to_make_available():
        return "Give a look at 'https://pynput.readthedocs.io/en/latest/limitations.html' on how to allow the use of the keyboard listener."
