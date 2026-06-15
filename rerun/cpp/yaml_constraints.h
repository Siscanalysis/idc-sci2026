// yaml_constraints.h — load `constraints:` from a problem YAML and apply them
// to an opennn::ResponseOptimization instance. Lets each run_idc_*.cpp share
// one source of truth with its problem YAML, so editing the YAML alone
// suffices to change the optimized problem on both pymoo and IDC sides.
//
// Supports two constraint shapes:
//   - linear_inequality   (with `coefficients` + `lower`/`upper`)
//   - nonlinear_inequality (with `expression`, pymoo convention g(x) <= 0)
//
// The YAML schema this parses is the narrow one used by IDC_benchmark/problems/:
//
//   constraints:
//     - name: ...
//       type: linear_inequality
//       coefficients: { var: w, ... }     # or multi-line `key: value` mapping
//       lower: number                     # optional
//       upper: number                     # optional
//     - { name: ..., type: nonlinear_inequality, expression: "..." }
//
// Python-style `**` in expressions is rewritten to opennn's `^` operator.

#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "response_optimization.h"

namespace opennn_idc {

// Coefficients are stored as an ordered list (not a map) so that the
// generated formula string preserves the YAML's variable order. Some
// ResponseOptimization internals are sensitive to expression token order,
// so reordering would silently change seed-level results.
struct ConstraintSpec
{
    std::string name;
    std::string type;
    std::vector<std::pair<std::string, float>> coefficients;
    std::string expression;
    bool has_lower = false;
    bool has_upper = false;
    float lower = 0.0f;
    float upper = 0.0f;
};

namespace detail {

inline std::string strip(std::string s)
{
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    if(a == std::string::npos) return "";
    return s.substr(a, b - a + 1);
}

inline std::string strip_quotes(std::string s)
{
    if(s.size() >= 2 && ((s.front() == '"'  && s.back() == '"') ||
                         (s.front() == '\'' && s.back() == '\'')))
        return s.substr(1, s.size() - 2);
    return s;
}

// Strip a trailing `# comment` from a YAML value, taking care not to eat
// `#` characters that sit inside a quoted string. We only honor quotes when
// the value STARTS with one — sufficient for the IDC YAML schema.
inline std::string strip_inline_comment(std::string s)
{
    s = strip(s);
    if(!s.empty() && (s.front() == '"' || s.front() == '\''))
    {
        char q = s.front();
        size_t end = s.find(q, 1);
        if(end == std::string::npos) return s;
        return s.substr(0, end + 1);
    }
    size_t hash = s.find('#');
    if(hash == std::string::npos) return s;
    return strip(s.substr(0, hash));
}

inline int indent_of(const std::string& line)
{
    int n = 0;
    while(n < (int)line.size() && (line[n] == ' ' || line[n] == '\t'))
        ++n;
    return n;
}

inline bool blank_or_comment(const std::string& line)
{
    for(char c : line)
    {
        if(c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
        return c == '#';
    }
    return true;
}

// Parse "{ a: 1, b: 2 }" or "a: 1, b: 2" into an ordered list of
// (name, weight) pairs. Order from the YAML is preserved.
// Naive comma split — fine for our schema (no commas inside expressions).
inline std::vector<std::pair<std::string, float>> parse_inline_dict(std::string body)
{
    body = strip(body);
    if(!body.empty() && body.front() == '{') body.erase(body.begin());
    if(!body.empty() && body.back()  == '}') body.pop_back();

    std::vector<std::pair<std::string, float>> out;
    std::string token;
    std::istringstream ss(body);
    while(std::getline(ss, token, ','))
    {
        size_t colon = token.find(':');
        if(colon == std::string::npos) continue;
        std::string k = strip(token.substr(0, colon));
        std::string v = strip(token.substr(colon + 1));
        if(k.empty() || v.empty()) continue;
        out.emplace_back(std::move(k), std::stof(v));
    }
    return out;
}

// Rewrite Python/YAML `**` power to opennn's `^`.
inline std::string normalize_expression(std::string e)
{
    std::string out;
    out.reserve(e.size());
    for(size_t i = 0; i < e.size(); ++i)
    {
        if(i + 1 < e.size() && e[i] == '*' && e[i+1] == '*')
        {
            out.push_back('^');
            ++i;
        }
        else out.push_back(e[i]);
    }
    return out;
}

// Build "w1*var1 + w2*var2 + ...". Weights of 0 dropped; weights of ±1
// emit "var" / "-var". The first term is emitted without a leading "+".
inline std::string build_linear_expr(const std::vector<std::pair<std::string, float>>& coeffs)
{
    std::ostringstream out;
    bool first = true;
    for(const auto& [name, w] : coeffs)
    {
        if(w == 0.0f) continue;
        if(first)
        {
            if(w == 1.0f)  out << name;
            else if(w == -1.0f) out << "-" << name;
            else out << w << "*" << name;
            first = false;
        }
        else if(w > 0.0f)
        {
            out << " + ";
            if(w == 1.0f) out << name;
            else out << w << "*" << name;
        }
        else
        {
            out << " - ";
            if(w == -1.0f) out << name;
            else out << (-w) << "*" << name;
        }
    }
    if(first) out << "0";
    return out.str();
}

}   // namespace detail

inline std::vector<ConstraintSpec> parse_yaml_constraints(const std::filesystem::path& yaml_path)
{
    using namespace detail;

    std::ifstream f(yaml_path);
    if(!f)
    {
        // A missing IDC problem YAML is treated as "no constraints" — the
        // driver still runs, optimizing only box bounds + the objective.
        // Common case: newly generated drivers that haven't yet had a
        // problems/<name>.yaml hand-authored for them. Loud-warn so this
        // can't go unnoticed in the log.
        std::cerr << "[yaml-constraints] WARN: " << yaml_path.string()
                  << " not found; running with no formula constraints.\n";
        return {};
    }

    std::vector<std::string> lines;
    std::string line;
    while(std::getline(f, line)) lines.push_back(line);

    // Locate top-level "constraints:" key (indent == 0).
    int header_indent = -1;
    size_t start = 0;
    for(size_t i = 0; i < lines.size(); ++i)
    {
        if(blank_or_comment(lines[i])) continue;
        if(indent_of(lines[i]) != 0) continue;
        std::string body = strip(lines[i]);
        if(body.rfind("constraints:", 0) == 0)
        {
            header_indent = 0;
            start = i + 1;
            break;
        }
    }
    if(header_indent < 0) return {};

    std::vector<ConstraintSpec> specs;

    auto consume_kv = [](ConstraintSpec& spec, const std::string& key, const std::string& raw_val)
    {
        std::string v = strip_quotes(strip_inline_comment(raw_val));
        if(key == "name") spec.name = v;
        else if(key == "type") spec.type = v;
        else if(key == "expression") spec.expression = v;
        else if(key == "lower") { spec.has_lower = true; spec.lower = std::stof(v); }
        else if(key == "upper") { spec.has_upper = true; spec.upper = std::stof(v); }
        else if(key == "coefficients")
        {
            // Inline `{ ... }` form on the same line. Multi-line dict is handled
            // by the caller (it sees `coefficients:` with an empty value).
            if(!v.empty() && v.front() == '{')
                spec.coefficients = parse_inline_dict(v);
        }
    };

    size_t i = start;
    while(i < lines.size())
    {
        if(blank_or_comment(lines[i])) { ++i; continue; }
        int ind = indent_of(lines[i]);
        if(ind <= header_indent) break;

        std::string body = strip(lines[i]);
        if(body.rfind("- ", 0) != 0 && body != "-") { ++i; continue; }

        ConstraintSpec spec;
        int entry_indent = ind;
        std::string after = (body.size() > 1) ? strip(body.substr(1)) : "";

        // ----- Case A: `- { name: g1, type: ..., expression: "..." }` ------
        if(!after.empty() && after.front() == '{')
        {
            std::string m = after;
            m.erase(m.begin());
            if(!m.empty() && m.back() == '}') m.pop_back();

            std::string tok;
            std::istringstream ss(m);
            while(std::getline(ss, tok, ','))
            {
                size_t colon = tok.find(':');
                if(colon == std::string::npos) continue;
                std::string k = strip(tok.substr(0, colon));
                std::string v = strip(tok.substr(colon + 1));
                consume_kv(spec, k, v);
            }
            specs.push_back(std::move(spec));
            ++i;
            continue;
        }

        // ----- Case B: multi-line `- name: ...\n  type: ...\n  ...` --------
        if(!after.empty())
        {
            size_t colon = after.find(':');
            if(colon != std::string::npos)
                consume_kv(spec, strip(after.substr(0, colon)), strip(after.substr(colon + 1)));
        }

        ++i;
        while(i < lines.size())
        {
            if(blank_or_comment(lines[i])) { ++i; continue; }
            int cind = indent_of(lines[i]);
            if(cind <= entry_indent) break;

            std::string cbody = strip(lines[i]);
            size_t colon = cbody.find(':');
            if(colon == std::string::npos) { ++i; continue; }

            std::string k = strip(cbody.substr(0, colon));
            std::string v = strip(cbody.substr(colon + 1));

            if(k == "coefficients" && v.empty())
            {
                int kind = cind;
                ++i;
                while(i < lines.size())
                {
                    if(blank_or_comment(lines[i])) { ++i; continue; }
                    int kkind = indent_of(lines[i]);
                    if(kkind <= kind) break;
                    std::string kb = strip(lines[i]);
                    size_t kc = kb.find(':');
                    if(kc == std::string::npos) { ++i; continue; }
                    spec.coefficients.emplace_back(strip(kb.substr(0, kc)),
                                                   std::stof(strip(kb.substr(kc + 1))));
                    ++i;
                }
                continue;
            }

            consume_kv(spec, k, v);
            ++i;
        }

        specs.push_back(std::move(spec));
    }

    return specs;
}

// Apply parsed specs to a ResponseOptimization. Logs one line per constraint.
inline void apply_yaml_constraints(opennn::ResponseOptimization& opt,
                                   const std::vector<ConstraintSpec>& specs)
{
    using Op = opennn::ResponseOptimization::ConditionType;

    for(const auto& s : specs)
    {
        std::string expr;
        Op op = Op::None;
        float lo = 0.0f, hi = 0.0f;

        if(s.type == "linear_inequality")
        {
            expr = detail::build_linear_expr(s.coefficients);
            if(s.has_lower && s.has_upper)
            {
                if(s.lower == s.upper) { op = Op::EqualTo; lo = hi = s.upper; }
                else                    { op = Op::Between; lo = s.lower; hi = s.upper; }
            }
            else if(s.has_lower) { op = Op::GreaterEqualTo; lo = hi = s.lower; }
            else if(s.has_upper) { op = Op::LessEqualTo;    lo = hi = s.upper; }
            else throw std::runtime_error("linear_inequality '" + s.name +
                                          "' has neither lower nor upper bound.");
        }
        else if(s.type == "nonlinear_inequality")
        {
            if(s.expression.empty())
                throw std::runtime_error("nonlinear_inequality '" + s.name +
                                         "' has empty expression.");
            expr = detail::normalize_expression(s.expression);
            op = Op::LessEqualTo;
            lo = hi = 0.0f;
        }
        else
        {
            throw std::runtime_error("Unknown constraint type: " + s.type);
        }

        opt.set_formula_constraint(expr, op, lo, hi);

        std::cout << "  [yaml " << s.name << "] " << expr;
        switch(op)
        {
            case Op::EqualTo:        std::cout << " == " << lo; break;
            case Op::Between:        std::cout << " in [" << lo << ", " << hi << "]"; break;
            case Op::LessEqualTo:    std::cout << " <= " << hi; break;
            case Op::GreaterEqualTo: std::cout << " >= " << lo; break;
            default: break;
        }
        std::cout << "\n";
    }
}

inline void apply_yaml_constraints(opennn::ResponseOptimization& opt,
                                   const std::filesystem::path& yaml_path)
{
    apply_yaml_constraints(opt, parse_yaml_constraints(yaml_path));
}

}   // namespace opennn_idc
