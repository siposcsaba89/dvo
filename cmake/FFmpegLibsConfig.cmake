# Imported targets FFmpeg::<component> for the FFmpeg libraries (vcpkg or system install, Windows and Linux).
#   find_package(FFmpegLibs CONFIG REQUIRED COMPONENTS avcodec avutil swscale PATHS ${CMAKE_SOURCE_DIR}/cmake)
# The package is not called FFmpeg: vcpkg redirects find_package(FFmpeg) to its own variable-based wrapper.
include(FindPackageHandleStandardArgs)

if(NOT FFmpegLibs_FIND_COMPONENTS)
  set(FFmpegLibs_FIND_COMPONENTS avcodec avutil)
endif()
# Link order: dependants first.
set(_ffmpeg_deps_avcodec avutil)
set(_ffmpeg_deps_avformat avcodec avutil)
set(_ffmpeg_deps_swscale avutil)
set(_ffmpeg_deps_swresample avutil)
set(_ffmpeg_deps_avfilter avformat avcodec swscale swresample avutil)
set(_ffmpeg_deps_avdevice avformat avfilter avcodec avutil)

set(_ffmpeg_components ${FFmpegLibs_FIND_COMPONENTS})
foreach(_c IN LISTS FFmpegLibs_FIND_COMPONENTS)
  list(APPEND _ffmpeg_components ${_ffmpeg_deps_${_c}})
endforeach()
list(REMOVE_DUPLICATES _ffmpeg_components)

find_path(FFmpegLibs_INCLUDE_DIR libavcodec/avcodec.h)

# vcpkg keeps debug builds under <prefix>/debug/lib with the same names, and its toolchain may put that directory on
# the search path of multi-config generators; the release library is looked up in <prefix>/lib first.
get_filename_component(_ffmpeg_prefix "${FFmpegLibs_INCLUDE_DIR}" DIRECTORY)
foreach(_c IN LISTS _ffmpeg_components)
  find_library(FFmpegLibs_${_c}_LIBRARY_RELEASE NAMES ${_c} lib${_c} PATHS "${_ffmpeg_prefix}/lib" NO_DEFAULT_PATH
    NO_CACHE)
  if(NOT FFmpegLibs_${_c}_LIBRARY_RELEASE)
    find_library(FFmpegLibs_${_c}_LIBRARY_RELEASE NAMES ${_c} lib${_c} NO_CACHE)
  endif()
  find_library(FFmpegLibs_${_c}_LIBRARY_DEBUG NAMES ${_c} lib${_c} HINTS "${_ffmpeg_prefix}/debug/lib"
    NO_DEFAULT_PATH NO_CACHE)
  if(FFmpegLibs_${_c}_LIBRARY_RELEASE AND FFmpegLibs_INCLUDE_DIR)
    set(FFmpegLibs_${_c}_FOUND TRUE)
  else()
    set(FFmpegLibs_${_c}_FOUND FALSE)
  endif()
endforeach()

find_package_handle_standard_args(FFmpegLibs REQUIRED_VARS FFmpegLibs_INCLUDE_DIR HANDLE_COMPONENTS)

if(FFmpegLibs_FOUND)
  foreach(_c IN LISTS _ffmpeg_components)
    if(NOT TARGET FFmpeg::${_c} AND FFmpegLibs_${_c}_FOUND)
      add_library(FFmpeg::${_c} UNKNOWN IMPORTED)
      set_target_properties(FFmpeg::${_c} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${FFmpegLibs_INCLUDE_DIR}"
        IMPORTED_CONFIGURATIONS RELEASE
        IMPORTED_LOCATION_RELEASE "${FFmpegLibs_${_c}_LIBRARY_RELEASE}"
        IMPORTED_LOCATION "${FFmpegLibs_${_c}_LIBRARY_RELEASE}"
        MAP_IMPORTED_CONFIG_MINSIZEREL Release
        MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
      if(FFmpegLibs_${_c}_LIBRARY_DEBUG)
        set_property(TARGET FFmpeg::${_c} APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
        set_target_properties(FFmpeg::${_c} PROPERTIES IMPORTED_LOCATION_DEBUG "${FFmpegLibs_${_c}_LIBRARY_DEBUG}")
      endif()
    endif()
  endforeach()
  foreach(_c IN LISTS _ffmpeg_components)
    foreach(_d IN LISTS _ffmpeg_deps_${_c})
      if(TARGET FFmpeg::${_d})
        set_property(TARGET FFmpeg::${_c} APPEND PROPERTY INTERFACE_LINK_LIBRARIES FFmpeg::${_d})
      endif()
    endforeach()
  endforeach()
endif()
