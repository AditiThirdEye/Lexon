# Lexon Compiler BERT Emotion Classification Example

## Introduction
This example shows how to use Lexon Compiler to compile a BERT model to MLIR code then run it.  The [model](bhadresh-savani/bert-base-uncased-emotion) is trained to classify the emotion of a sentence into one of the following classes: sadness, joy,  love, anger, fear, and surprise.


## How to run
1. Ensure that LLVM, Lexon Compiler and the Lexon Compiler python packages are installed properly. You can refer to [here](https://github.com/lexon-compiler/lexon-mlir) for more information and do a double check.

2. Set the `PYTHONPATH` environment variable.
```bash
$ export PYTHONPATH=/path-to-lexon-mlir/llvm/build/tools/mlir/python_packages/mlir_core:/path-to-lexon-mlir/build/python_packages:${PYTHONPATH}
```

3. Build and run the BERT example
```bash
$ cmake -G Ninja .. -DLEXON_BERT_EXAMPLES=ON
$ ninja lexon-bert-run
$ cd bin
$ ./lexon-bert-run
```

4. Enjoy it!
