cmake_minimum_required(VERSION 3.13)

include(pico_sdk_import.cmake)

project(mnist_edge_ai C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(PICO_BOARD pico CACHE STRING "Board type")

pico_sdk_init()

add_subdirectory(pico-tflmicro EXCLUDE_FROM_ALL)

add_executable(mnist_edge_ai
    main.cpp
    mnist_model_data.cpp
)

target_link_libraries(mnist_edge_ai
    pico_stdlib
    pico-tflmicro
)

pico_enable_stdio_usb(mnist_edge_ai 1)
pico_enable_stdio_uart(mnist_edge_ai 0)

pico_add_extra_outputs(mnist_edge_ai)