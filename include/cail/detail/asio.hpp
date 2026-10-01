#pragma once

#if defined(GLZ_USE_BOOST_ASIO) || !__has_include(<asio.hpp>)
#include <boost/asio.hpp>
namespace cail {
namespace asio = boost::asio;
}
#else
#include <asio.hpp>
namespace cail {
namespace asio = ::asio;
}
#endif
