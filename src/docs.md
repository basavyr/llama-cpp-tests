# Building and compile custom C++ with `llama.h`

## Dynamic vs static

In order to compile `.c++` files that require usage of the libraries from `llama.cpp`, a build with static libraries is required. This is to prevent issues while trying to compile sources with functions that require some libraries that are only available as `.dylib` (i.e., dynamically built). An example with dynamic libraries can be seen below, where the project [was built](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) with no additional flags:
```bash
build/bin/libllama.0.dylib
build/bin/libmtmd.0.dylib
build/bin/libggml.0.dylib
build/bin/libllama.dylib
build/bin/libmtmd.dylib
build/bin/libggml.dylib
```
As it can be seen, all libraries are `.dylib`, making the compilation of other sources that will include these dependencies quite cumbersome.

In order to **build `llama.cpp` with static libraries**, the following commands are necessary:
```bash
cmake -B build -DBUILD_SHARED_LIBS=OFF
cmake --build build --config Release -j 8
```

Once `llama.cpp` is available in such format, one can check if all the important libraries such as `ggml` and `llama` are available as `.a` files.
```bash
build/common/libcommon.a
build/src/libllama.a
build/tools/mtmd/libmtmd.a
build/ggm1/src/libggml.a
```

## Test the static build

After the static build has been finished, it can be tested using a source file that is also available within `llama.cpp`. As an example, one can take `examples/simple` case, assuming that the current working directory is the root of `llama.cpp` repository itself.

1. From `llama.cpp/`:
```bash
g++ examples/simple/simple.cpp \
    -std=c++17 \
    -Iinclude \
    -Icommon \
    -Iggml/include \
    -Ibuild/common \
    \
    -Lbuild/common \
    -Lbuild/src \
    -Lbuild/ggml/src \
    -Lbuild/ggml/src/ggml-metal \
    -Lbuild/ggml/src/ggml-blas \
    \
    -lcommon \
    -lllama \
    -lggml \
    -lggml-base \
    -lggml-cpu \
    -lggml-metal \
    -lggml-blas \
    \
    -framework Accelerate \
    -framework Foundation \
    -framework Metal \
    -framework MetalKit \
    -o my_simple
```
2. This will create a new executable, `my-simple` (i.e., the `$TARGET`):
```bash
ls -p | grep -v /
AUTHORS
CMakeLists.txt
CMakePresets.json
CODEOWNERS
CONTRIBUTING.md
LICENSE
Makefile
README.md
SECURITY.md
build-xcframework.sh
convert_hf_to_gguf.py
convert_hf_to_gguf_update.py
convert_llama_ggml_to_gguf.py
convert_lora_to_gguf.py
flake.lock
flake.nix
my_simple     <----------- the created TARGET
mypy.ini
poetry.lock
pyproject.toml
pyrightconfig.json
requirements.txt
```
3. Running the executable will provide the proper workflow of the implementation. The source code of `simple` can be seen [here](https://github.com/ggml-org/llama.cpp/tree/master/examples/simple).
```bash
./my_simple 

example usage:

    ./my_simple -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]
```

## Compile with `llama.cpp` libraries from other locations

The current scenario requires compilation of source code that will use `llama.cpp` functions from outside its root directory, as this repository will pe located elsewhere. In order to make compilation work, **include paths** and also **linking paths** should be properly provided.

For completeness, everything is defined in the available [`Makefile`](./Makefile). In this file, **the only variable that should require some adjustments is the `ROOT_DIR`**, which should point to the statically built `llama.cpp` project that was mentioned above.

1. The **headers** are provided via:
```makefile
INCLUDES = -I$(ROOT_DIR)/include \
           -I$(ROOT_DIR)/common \
           -I$(ROOT_DIR)/ggml/include \
           -I$(BUILD_DIR)/common
```
2. The **library paths** can be provided with:
```makefile
LIB_PATHS = -L$(BUILD_DIR)/src \
            -L$(BUILD_DIR)/common \
            -L$(BUILD_DIR)/ggml/src \
            -L$(BUILD_DIR)/ggml/src/ggml-metal \
            -L$(BUILD_DIR)/ggml/src/ggml-blas
```
3. The linked libraries can be provided by:
```makefile
LIBS = -lcommon \
       -lllama \
       -lggml \
       -lggml-base \
       -lggml-cpu \
       -lggml-metal \
       -lggml-blas
```
4. Lastly, the `ggml` implementations do require several frameworks:
```makefile
FRAMEWORKS = -framework Accelerate \
             -framework Foundation \
             -framework Metal \
             -framework MetalKit
```

With all these handled within the Makefile, execution of `make` should provide the following output:
```bash
make
g++  -I/Users/svc_sps/Documents/External-GitHub/static-llama/llama.cpp/include -I/Users/svc_sps/Documents/External-GitHub/static-llama/llama.cpp/common -I/Users/svc_sps/Documents/External-GitHub/static-llama/llama.cpp/ggml/include -I/Users/svc_sps/Documents/External-GitHub/static-llama/llama.cpp/build/common -L/Users/svc_sps/Documents/External-GitHub/static-llama/llama.cpp/build/src -L/Users/svc_sps/Documents/External-GitHub/static-llama/llama.cpp/build/common -L/Users/svc_sps/Documents/External-GitHub/static-llama/llama.cpp/build/ggml/src -L/Users/svc_sps/Documents/External-GitHub/static-llama/llama.cpp/build/ggml/src/ggml-metal -L/Users/svc_sps/Documents/External-GitHub/static-llama/llama.cpp/build/ggml/src/ggml-blas simple-test.cpp -lcommon -lllama -lggml -lggml-base -lggml-cpu -lggml-metal -lggml-blas -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -o simple-test.o
```