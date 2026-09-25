#pragma once

#include <cstdlib>
#include <string>

namespace cail::detail {

[[nodiscard]] inline std::string env_or(std::string value, const char* variable)
{
    if (!value.empty() || variable == nullptr) {
        return value;
    }
    if (const char* env = std::getenv(variable); env != nullptr && env[0] != '\0') {
        return env;
    }
    return value;
}

} // namespace cail::detail
