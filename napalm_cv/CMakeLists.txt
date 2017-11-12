cmake_minimum_required(VERSION 3.0)

project(napalm_cv)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

option(BUILD_EXAMPLES "Build cv examples" ON)
if (NOT TARGET ::napalm)
    find_package(napalm REQUIRED)
endif()

set(PUBLIC_HDRS
    include/napalm/cv/cv.h
)

set(SRCS 
    src/cv.cpp
)

add_library(${PROJECT_NAME} ${PUBLIC_HDRS} ${SRCS})
set (PROGRAMS src/convolv.cl)
set(EMBED_PROGRAM_FILES FALSE)
set(COMPILE_IN_BUILD_TYPE_IF_POSSIBLE FALSE)
set(PROGRAM_RUNTIME_SOURCE_DIR "./kernels/")
create_programs("${PROGRAMS}"
    output ${PROJECT_NAME}
    COMPILE_IN_BUILD_TYPE_IF_POSSIBLE 
    ${EMBED_PROGRAM_FILES}
    ${PROGRAM_RUNTIME_SOURCE_DIR}
)


add_library(::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
if (BUILD_SHARED_LIBS)
    set_target_properties(${PROJECT_NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
    set_target_properties(${PROJECT_NAME} PROPERTIES VISIBILITY_INLINES_HIDDEN 1)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX _d)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER libs)
if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE "/MP")
endif()

source_group("napalm\\cv" FILES ${PUBLIC_HDRS})
source_group("src" FILES ${SRCS})

target_link_libraries(${PROJECT_NAME} PUBLIC ::napalm)

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME} EXPORT_FILE_NAME ${CMAKE_BINARY_DIR}/gen/napalm/cv/${PROJECT_NAME}_export.h)


target_include_directories(${PROJECT_NAME} 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/gen>
        $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/gen/${PROJECT_NAME}-config-version.cmake"
  VERSION 0.0.1
  COMPATIBILITY AnyNewerVersion
)

configure_file(cmake/${PROJECT_NAME}-config.cmake.in ${PROJECT_NAME}-config.cmake @ONLY)
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets DESTINATION 
    ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(FILES 
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake 
        ${CMAKE_CURRENT_BINARY_DIR}/gen/${PROJECT_NAME}-config-version.cmake
    DESTINATION 
        lib/cmake/${PROJECT_NAME})
    
install(EXPORT ${PROJECT_NAME}-targets NAMESPACE :: DESTINATION lib/cmake/${PROJECT_NAME})

install(FILES
    ${CMAKE_BINARY_DIR}/gen/napalm/cv/${PROJECT_NAME}_export.h
	${PUBLIC_HDRS}
    DESTINATION include/${PROJECT_NAME}/napalm/cv)

if (BUILD_EXAMPLES)
    add_subdirectory(samples/tests)
endif()
    
# uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
    set_target_properties(uninstall PROPERTIES FOLDER CMakePredefinedTargets)
endif()
    