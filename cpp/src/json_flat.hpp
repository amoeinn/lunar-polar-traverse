// Minimal reader and writer for a flat JSON object: string, number and boolean
// values only, no nesting and no arrays. The schema is deliberately flat so
// that this parser is small enough to be read and checked by eye; the model's
// parameter file and sidecar are the only things that ever pass through it.
#pragma once
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace json_flat {

class Object {
public:
    std::map<std::string, std::string> strings;
    std::map<std::string, double> numbers;
    std::map<std::string, bool> bools;

    const std::string& str(const std::string& k) const {
        auto it = strings.find(k);
        if (it == strings.end()) throw std::runtime_error("missing string key: " + k);
        return it->second;
    }
    double num(const std::string& k) const {
        auto it = numbers.find(k);
        if (it == numbers.end()) throw std::runtime_error("missing number key: " + k);
        return it->second;
    }
    bool has_str(const std::string& k) const {
        return strings.find(k) != strings.end();
    }
    double num(const std::string& k, double fallback) const {
        auto it = numbers.find(k);
        return it == numbers.end() ? fallback : it->second;
    }
    int integer(const std::string& k) const {
        double v = num(k);
        if (v != static_cast<double>(static_cast<long long>(v)))
            throw std::runtime_error("key is not an integer: " + k);
        return static_cast<int>(v);
    }
};

namespace detail {

inline void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
}

inline std::string parse_string(const std::string& s, size_t& i) {
    if (s[i] != '"') throw std::runtime_error("expected a string");
    ++i;
    std::string out;
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\') {
            ++i;
            if (i >= s.size()) throw std::runtime_error("truncated escape");
            switch (s[i]) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case '/': out += '/';  break;
                case '"': out += '"';  break;
                case '\\': out += '\\'; break;
                default: throw std::runtime_error("unsupported escape");
            }
            ++i;
        } else {
            out += s[i++];
        }
    }
    if (i >= s.size()) throw std::runtime_error("unterminated string");
    ++i;
    return out;
}

}  // namespace detail

inline Object parse(const std::string& text) {
    Object o;
    size_t i = 0;
    detail::skip_ws(text, i);
    if (i >= text.size() || text[i] != '{') throw std::runtime_error("expected '{'");
    ++i;
    detail::skip_ws(text, i);
    if (i < text.size() && text[i] == '}') return o;
    while (true) {
        detail::skip_ws(text, i);
        std::string key = detail::parse_string(text, i);
        detail::skip_ws(text, i);
        if (i >= text.size() || text[i] != ':') throw std::runtime_error("expected ':'");
        ++i;
        detail::skip_ws(text, i);
        if (i >= text.size()) throw std::runtime_error("truncated value");
        if (text[i] == '"') {
            o.strings[key] = detail::parse_string(text, i);
        } else if (text.compare(i, 4, "true") == 0) {
            o.bools[key] = true; i += 4;
        } else if (text.compare(i, 5, "false") == 0) {
            o.bools[key] = false; i += 5;
        } else if (text.compare(i, 4, "null") == 0) {
            i += 4;
        } else {
            size_t start = i;
            while (i < text.size() && (std::isdigit(static_cast<unsigned char>(text[i])) ||
                                       text[i] == '-' || text[i] == '+' || text[i] == '.' ||
                                       text[i] == 'e' || text[i] == 'E')) ++i;
            if (start == i) throw std::runtime_error("expected a value for key: " + key);
            o.numbers[key] = std::stod(text.substr(start, i - start));
        }
        detail::skip_ws(text, i);
        if (i < text.size() && text[i] == ',') { ++i; continue; }
        if (i < text.size() && text[i] == '}') { ++i; break; }
        throw std::runtime_error("expected ',' or '}'");
    }
    return o;
}

inline Object load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return parse(ss.str());
}

class Writer {
public:
    void add(const std::string& k, const std::string& v) { emit(k, "\"" + escape(v) + "\""); }
    // Without this overload a string literal binds to add(..., bool), because
    // const char* -> bool is a standard conversion and const char* -> std::string
    // is a user-defined one. Every string in the sidecar then came out as `true`.
    void add(const std::string& k, const char* v) { add(k, std::string(v)); }
    void add(const std::string& k, double v) {
        std::ostringstream os;
        os.precision(17);
        os << v;
        emit(k, os.str());
    }
    void add(const std::string& k, long long v) { emit(k, std::to_string(v)); }
    void add(const std::string& k, int v) { emit(k, std::to_string(v)); }
    void add(const std::string& k, bool v) { emit(k, v ? "true" : "false"); }

    void save(const std::string& path) const {
        std::ofstream f(path);
        if (!f) throw std::runtime_error("cannot write " + path);
        f << "{\n" << body_ << "\n}\n";
    }

private:
    static std::string escape(const std::string& v) {
        std::string out;
        for (char c : v) {
            if (c == '"' || c == '\\') { out += '\\'; out += c; }
            else if (c == '\n') out += "\\n";
            else out += c;
        }
        return out;
    }
    void emit(const std::string& k, const std::string& v) {
        if (!body_.empty()) body_ += ",\n";
        body_ += "  \"" + escape(k) + "\": " + v;
    }
    std::string body_;
};

}  // namespace json_flat
