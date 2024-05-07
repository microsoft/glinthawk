# Glinthawk

An inference engine for the Llama2 model family, written in C++.

## Development

### Dependencies

- [CMake >=3.18](https://cmake.org/)
- [GCC >=12](https://gcc.gnu.org/) (C++20 support required)
- [OpenSSL](https://www.openssl.org/)
- [Protobuf](https://developers.google.com/protocol-buffers)
- [Google Logging Library](https://github.com/google/glog)

For building the CUDA version, you will also need:
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

Make sure your `nvcc` is compatible with your GCC version.

### Building

```bash
mkdir build
cd build
cmake ..
make -j`nproc`
```

Please adjust according to your setup as needed. Tested only on Ubuntu 22.04
and later. This program requires C++20 support, and makes use of some
Linux-specific system calls (like `memfd_create`), although in a limited way.

For more information on building this project, please take a look at the
[Dockerfile.amd64](docker/Dockerfile.amd64) and
[Dockerfile.cuda](docker/Dockerfile.cuda) files.

### Test Models

For testing purposes, you can use
[tinyllamas](https://huggingface.co/karpathy/tinyllamas). Please use the
`tools/bin2glint.py` script to convert `.bin` files to Glinthawk's format. There
are other scripts in the `tools` directory for converting the original Llama2
models.

## Trademark Notice

Trademarks This project may contain trademarks or logos for projects, products,
or services. Authorized use of Microsoft trademarks or logos is subject to and
must follow Microsoft’s Trademark & Brand Guidelines. Use of Microsoft
trademarks or logos in modified versions of this project must not cause
confusion or imply Microsoft sponsorship. Any use of third-party trademarks or
logos are subject to those third-party’s policies.
