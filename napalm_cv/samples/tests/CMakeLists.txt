cmake_minimum_required(VERSION 3.0)

project(napalm_cv_tests)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if (NOT TARGET ::napalm_cv)
    find_package(napalm_cv REQUIRED)
endif()

add_executable(${PROJECT_NAME} main.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX _d)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER examples)
if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE "/MP")
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE ::napalm_cv)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)

