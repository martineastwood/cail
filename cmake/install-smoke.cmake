execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${CAIL_BUILD_DIR}" --prefix "${CAIL_INSTALL_PREFIX}"
        --config "${CAIL_CONFIGURATION}"
    RESULT_VARIABLE install_result
)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "Installing CAIL for the package smoke test failed.")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${CAIL_SOURCE_DIR}/tests/consumer" -B "${CAIL_TEST_BUILD}"
        -G "${CAIL_GENERATOR}"
        "-DCMAKE_PREFIX_PATH=${CAIL_INSTALL_PREFIX}"
    RESULT_VARIABLE configure_result
)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "Configuring the CAIL package consumer failed.")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${CAIL_TEST_BUILD}" --config "${CAIL_CONFIGURATION}"
    RESULT_VARIABLE build_result
)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "Building the CAIL package consumer failed.")
endif()

execute_process(
    COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${CAIL_TEST_BUILD}" --output-on-failure
        -C "${CAIL_CONFIGURATION}"
    RESULT_VARIABLE test_result
)
if(NOT test_result EQUAL 0)
    message(FATAL_ERROR "Running the CAIL package consumer failed.")
endif()
