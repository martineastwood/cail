#pragma once

#include <string>
#include <string_view>
#include <utility>

namespace cail::detail {

struct ServerSentEvent {
    std::string event;
    std::string data;
};

class SseParser {
    public:
    template <typename Handler> void feed(std::string_view bytes, Handler&& on_event)
    {
        for (const char byte : bytes) {
            if (byte == '\r') {
                process_line(on_event);
                skip_lf_ = true;
            } else if (byte == '\n') {
                if (skip_lf_) {
                    skip_lf_ = false;
                } else {
                    process_line(on_event);
                }
            } else {
                skip_lf_ = false;
                line_.push_back(byte);
            }
        }
    }

    template <typename Handler> void finish(Handler&& on_event)
    {
        if (!line_.empty()) {
            process_line(on_event);
        }
        dispatch(on_event);
    }

    private:
    template <typename Handler> void process_line(Handler& on_event)
    {
        if (line_.empty()) {
            dispatch(on_event);
        } else if (line_.front() != ':') {
            const auto colon = line_.find(':');
            const auto field = std::string_view{line_}.substr(0, colon);
            auto value = colon == std::string::npos ? std::string_view{} : std::string_view{line_}.substr(colon + 1);
            if (!value.empty() && value.front() == ' ') {
                value.remove_prefix(1);
            }
            if (field == "data") {
                data_.append(value);
                data_.push_back('\n');
            } else if (field == "event") {
                event_.assign(value);
            }
        }
        line_.clear();
    }

    template <typename Handler> void dispatch(Handler& on_event)
    {
        if (data_.empty()) {
            event_.clear();
            return;
        }
        data_.pop_back();
        on_event(ServerSentEvent{.event = std::move(event_), .data = std::move(data_)});
        event_.clear();
        data_.clear();
    }

    std::string line_;
    std::string event_;
    std::string data_;
    bool skip_lf_{};
};

} // namespace cail::detail
