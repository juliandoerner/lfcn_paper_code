cmake_minimum_required(VERSION 3.0)
project(QM-Prototypes)

### needed for gcc to compile with c++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

### SETUP DEALII ###
FIND_PACKAGE(deal.II 9.4.0 QUIET
        HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
        )
IF (NOT ${deal.II_FOUND})
    MESSAGE(FATAL_ERROR "\n"
            "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
            "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
            "or set an environment variable \"DEAL_II_DIR\" that contains this path."
            )
ENDIF ()
DEAL_II_INITIALIZE_CACHED_VARIABLES()


# add dg-maxwell project
add_subdirectory(timaxdg)

# add project specific components

# add projects
add_subdirectory(src)