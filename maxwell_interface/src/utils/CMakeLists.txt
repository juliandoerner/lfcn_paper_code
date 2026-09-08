
add_library(utils
  Distort.h
  ExtractInterface.h
  GenerateDomain.h
  GenerateDomain.cpp
  ODESystem.h
  ODESystemOperators.h
  ODESystemOperators.cpp
  LFCN.h
  LFCN.hh
  LorentzSolution.h
  LorentzSolution.cpp
  PolynomialInterfaceSolution.h
  PolynomialInterfaceSolution.cpp)

target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(utils tools time_integration)

DEAL_II_SETUP_TARGET(utils)
