vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO martineastwood/cail
    REF "v${VERSION}"
    SHA512 a3d8ea1162e7c0a5c6e2fae03c632fd93ef12cc1a8fffaac3cc2011bce4dec495e64d490bd899e6f0c591e6cc4026e4ba70facdf825631c4f7c3fd582d057407
    HEAD_REF main
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DCAIL_BUILD_EXAMPLES=OFF
        -DBUILD_TESTING=OFF
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/cail)

# Header-only, so the debug tree duplicates the release tree.
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug")

# The package config moves to share/, leaving an empty lib/.
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
