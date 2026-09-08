
add_library(time_integration
        Leapfrog.h
        Leapfrog.hh
        CrankNicolson.h
        CrankNicolson.hh
        TimeIntegrator.h
        LSRK.h)
target_include_directories(time_integration PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(time_integration data_types)

DEAL_II_SETUP_TARGET(time_integration)