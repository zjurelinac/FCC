# FCC - _a FRISC C Compiler_

## Installation
Clone this repo, install python3, then type `pip install pycparser`, and you should be all set to go, if you're using Linux.
If, on the other hand, for some strange and uncomprehensible reason, you use Windows, you should also install **clang** (http://releases.llvm.org/download.html#3.9.1)

## Usage
Position yourself in the folder containing `compiler.py`, and type `python compiler.py "path_to_c_file"`. If you get some strange errors, 
perhaps you should try changing `python` to `python3` (this is common if you have multiple python versions installed). If the errors aren't
strange, then it's only you to blame, that is, the C file you tried to compile contains some errors - fix them and try again.

If everything goes as planned, you should now have an executable file of the same name next to the C source you compiled.
**That's all, folks!**
