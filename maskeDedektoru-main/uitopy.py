# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 12:39:15 2022

@author: User
"""

#.ui uzantılı dosyayı aşağıdaki kodla .py uzantısına dönüştürürüz.
#arayüz ile ilgili kısım
from PyQt5 import uic

with open('arayuzui.py', 'w', encoding="utf-8") as fout:
   uic.compileUi('arayuz.ui', fout)
