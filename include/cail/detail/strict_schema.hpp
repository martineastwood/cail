#pragma once

#include <cail/error.hpp>
#include <cail/schema.hpp>

#include <algorithm>
#include <map>
#include <memory>
#include <ranges>
#include <vector>

namespace cail::detail {

[[nodiscard]] inline bool has_schema_type(const Schema& schema, SchemaType type) {
    if (const auto* single_type = std::get_if<SchemaType>(&schema.type)) {
        return *single_type == type;
    }
    const auto& types = std::get<std::vector<SchemaType>>(schema.type);
    return std::ranges::find(types, type) != types.end();
}

inline void make_nullable(Schema& schema) {
    if (auto* single_type = std::get_if<SchemaType>(&schema.type)) {
        schema.type = std::vector<SchemaType>{*single_type, SchemaType::null};
        return;
    }
    auto& types = std::get<std::vector<SchemaType>>(schema.type);
    if (std::ranges::find(types, SchemaType::null) == types.end()) {
        types.emplace_back(SchemaType::null);
    }
}

[[nodiscard]] inline Result<Schema> strict_schema(const Schema& schema, bool root = true) {
    if (root) {
        const auto* type = std::get_if<SchemaType>(&schema.type);
        if (type == nullptr || *type != SchemaType::object) {
            return std::unexpected(Error{
                .code = ErrorCode::unsupported_schema,
                .message = "OpenAI strict structured output requires a root object schema.",
            });
        }
    }

    Schema result = schema;
    if (has_schema_type(schema, SchemaType::object)) {
        if (!schema.properties || !schema.required || !schema.additional_properties || *schema.additional_properties) {
            return std::unexpected(Error{
                .code = ErrorCode::unsupported_schema,
                .message = "OpenAI strict structured output requires object properties, a required list, and "
                           "additionalProperties=false.",
            });
        }

        for (const auto& name : *schema.required) {
            if (!schema.properties->contains(name)) {
                return std::unexpected(Error{
                    .code = ErrorCode::unsupported_schema,
                    .message = "The structured output schema requires an unknown property: " + name,
                });
            }
        }

        std::map<std::string, std::shared_ptr<Schema>> properties;
        std::vector<std::string> required;
        required.reserve(schema.properties->size());
        for (const auto& [name, child] : *schema.properties) {
            if (!child) {
                return std::unexpected(Error{
                    .code = ErrorCode::unsupported_schema,
                    .message = "The structured output schema contains a null property schema.",
                });
            }
            auto strict_child = strict_schema(*child, false);
            if (!strict_child) {
                return std::unexpected(strict_child.error());
            }
            if (std::ranges::find(*schema.required, name) == schema.required->end()) {
                make_nullable(*strict_child);
            }
            properties.emplace(name, std::make_shared<Schema>(std::move(*strict_child)));
            required.emplace_back(name);
        }
        result.properties = std::move(properties);
        result.required = std::move(required);
    }

    if (has_schema_type(schema, SchemaType::array)) {
        if (!schema.items) {
            return std::unexpected(Error{
                .code = ErrorCode::unsupported_schema,
                .message = "OpenAI structured output arrays require an items schema.",
            });
        }
        auto strict_items = strict_schema(*schema.items, false);
        if (!strict_items) {
            return std::unexpected(strict_items.error());
        }
        result.items = std::make_shared<Schema>(std::move(*strict_items));
    }

    return result;
}


} // namespace cail::detail
