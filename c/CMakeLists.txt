cmake_minimum_required(VERSION 3.16)
project(jkl-c VERSION 0.1.0 LANGUAGES C)

include(GNUInstallDirs)

option(JKL_BUILD_SHARED "Build shared jkl library" OFF)
option(JKL_BUILD_CLI "Build decode-only CLI tool" ON)

set(JKL_HEADERS
    include/jkl/ans.h
    include/jkl/bits.h
    include/jkl/elias.h
    include/jkl/error.h
    include/jkl/image.h
    include/jkl/jkli.h
    include/jkl/lz77.h
    include/jkl/rle.h
)

set(JKL_SOURCES
    src/ans.c
    src/bits.c
    src/elias.c
    src/image.c
    src/jkli.c
    src/lz77.c
    src/rle.c
)

if(JKL_BUILD_SHARED)
    add_library(jkl SHARED ${JKL_SOURCES} ${JKL_HEADERS})
else()
    add_library(jkl STATIC ${JKL_SOURCES} ${JKL_HEADERS})
endif()

add_library(jkl::jkl ALIAS jkl)

target_compile_features(jkl PUBLIC c_std_11)

target_include_directories(jkl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/jkl
)

if(MSVC)
    target_compile_options(jkl PRIVATE /W4)
else()
    target_compile_options(jkl PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(JKL_BUILD_CLI)
    add_executable(jkl-cli tools/cli.c)
    target_compile_features(jkl-cli PRIVATE c_std_11)
    target_include_directories(jkl-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/jkl)
    target_link_libraries(jkl-cli PRIVATE jkl)
endif()

install(TARGETS jkl
    EXPORT jklTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(JKL_BUILD_CLI)
    install(TARGETS jkl-cli
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

install(FILES ${JKL_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/jkl
)

install(EXPORT jklTargets
    NAMESPACE jkl::
    FILE jklTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jkl
)
