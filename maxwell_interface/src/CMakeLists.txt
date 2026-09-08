add_subdirectory(utils)

add_executable(InterfaceSolution_Prototype
        InterfaceSolution_Prototype.cpp)
target_link_libraries(InterfaceSolution_Prototype
        data
        data_types
        tools
        time_integration
        assembler_TE
        utils)
DEAL_II_SETUP_TARGET(InterfaceSolution_Prototype)

add_executable(InterfaceSolutionPolynomial_Prototype
        InterfaceSolutionPolynomial_Prototype.cpp)
target_link_libraries(InterfaceSolutionPolynomial_Prototype
        data
        data_types
        tools
        time_integration
        assembler_TE
        utils)
DEAL_II_SETUP_TARGET(InterfaceSolutionPolynomial_Prototype)
