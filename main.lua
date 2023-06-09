print("tcfvgbh")
-- main.lua
py = require 'python'

sum_from_python = py.import "test".test
print( sum_from_python() )