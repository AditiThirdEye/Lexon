# RISC-V Custom ISA Extension Examples

Currently, we support [Gemmini](https://github.com/ucb-bar/gemmini) as the custom extension in our backend.
To try these examples, please build [Gemmini](https://github.com/ucb-bar/gemmini#quick-start), enter the virtual environment, and run a makefile target.

For example:

```
// Complete the Gemmini build process.

$ cd /path/to/chipyard
$ source env.sh
$ cd lexon-mlir/examples/RISCVLexonExt
$ make rv-lexon-mvin-mvout-obj
$ make rv-lexon-mvin-mvout-exe
$ make rv-lexon-mvin-mvout-run
```
