 cmake_minimum_required(VERSION 3.0)

project(dvo-sample)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if (NOT TARGET ::napalm) 
    find_package(napalm REQUIRED) 
endif()
if (NOT TARGET ::napalm_cv) 
    find_package(napalm_cv REQUIRED) 
endif()
if (NOT TARGET ::dvo) 
    find_package(dvo REQUIRED) 
endif()

find_package(OpenCV REQUIRED)
set(SRCS 
    main.cpp
)

add_executable(${PROJECT_NAME} ${SRCS})
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX _d)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER examples)
if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE "/MP")
endif()
source_group("src" FILES ${SRCS})

target_link_libraries(${PROJECT_NAME} PRIVATE ::napalm ::dvo ::napalm_cv ${OpenCV_LIBRARIES}) 

install(TARGETS ${PROJECT_NAME} DESTINATION bin)