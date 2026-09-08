add_library(data STATIC
    CavitySolution.cpp
    PolynomialSolution.cpp
    include/IsotropicConstant.h)

target_include_directories(data PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(data tools)

DEAL_II_SETUP_TARGET(data)