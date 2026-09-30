#pragma once

#include <glaze/ext/glaze_asio.hpp>

namespace cail {
#if defined(GLZ_USING_BOOST_ASIO)
namespace asio = boost::asio;
#else
namespace asio = ::asio;
#endif
} // namespace cail
