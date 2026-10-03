// The executive: a thin behaviour tree that monitors one leg of one plan.
//
// It exists to test a consequence of the claim, not to be a mission executive.
// PLAN.md, "The executive", registers the tick definition, the
// replan-at-a-lit-cell constraint, k, the sample and the bars, all before this
// file was written.
//
// What it monitors is the continuous shadow budget and nothing else. Legality
// cannot change at execution time on static terrain and the core confirmed it,
// zero illegal steps in 187 plans; arrival time is not monitored because P3's
// clause 2 measured that the planners agree there.
//
// The world is the replay checker's trajectory dump: where the rover actually
// is, when, and whether it is lit. The monitor's forecast is the planner's own
// illumination model, deliberately, because the monitor is the planner side of
// the comparison and the checker is the independent judge. Where the two
// disagree the observation wins: if the world says dark where the model says
// lit, the darkness is projected forward to the model's next transition rather
// than the contradicted model being believed. That rule is what lets the
// monitor see a bust the planner's model could not predict, which is the only
// kind of bust this terrain produces.
//
// One leg, one tick stream. Chaining legs after a replan is the driver's job,
// because re-simulating the world is world simulation and not executive logic.
//
// Usage: executive <params.json>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "behaviortree_cpp/bt_factory.h"
#include "intervals_flat.hpp"
#include "json_flat.hpp"

namespace {

template <typename T>
std::vector<T> read_raw(const std::string& path, size_t n) {
    std::vector<T> v(n);
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(n * sizeof(T)));
    if (f.gcount() != static_cast<std::streamsize>(n * sizeof(T)))
        throw std::runtime_error("short read on " + path);
    return v;
}

template <typename T>
size_t count_of(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    return static_cast<size_t>(f.tellg()) / sizeof(T);
}

using ivflat::Intervals;

// One tick of the world: kind 0 departure, 1 cell arrival, 2 lit/dark
// transition, 3 end of a planned hold.
struct Tick {
    double t, shadow;
    int32_t index;
    uint8_t kind, lit;
};

struct Decision {
    int64_t tick;
    double t, s_exec, s_world, s_pred;
    int32_t index;
    uint8_t kind, lit, fired;
};

struct Ctx {
    // static for the leg
    Intervals iv;
    std::vector<int64_t> indptr;
    std::vector<int32_t> indices;
    std::vector<double> travel;
    std::vector<int32_t> plan;
    std::vector<double> plan_times;   // the planner's own schedule, for the slip estimate
    std::vector<double> plan_hold;    // the hold at each cell the schedule implies
    std::vector<Tick> world;
    double budget = 0.0;
    double vis_cadence_h = 1.0 / 60.0;
    int replan_limit = 2;
    double replan_cap_s = 30.0;
    std::string planner_bin, scratch, tmpl_params;

    // live
    size_t tick = 0;
    int pos = 0;
    double t_now = 0.0, t_prev = 0.0, s_exec = 0.0, s_world = 0.0, s_pred = 0.0;
    double t_leg_start = 0.0, planned_elapsed = 0.0;
    bool lit_now = true, have_lit = false;
    bool hold_pending = false;        // arrived at pos, planned hold not yet served
    int replans = 0;

    // outcome
    std::string outcome = "ran_to_end";   // or fired_replanned / fired_declared
    std::string reason;                   // declaration reason when declared
    double fired_at_t = std::numeric_limits<double>::quiet_NaN();
    int64_t fired_at_tick = -1;
    std::vector<int32_t> new_plan;
    std::vector<double> new_times;
    std::vector<Decision> log;

    double edge_travel(int a, int b) const {
        for (int64_t i = indptr[a]; i < indptr[a + 1]; ++i)
            if (indices[i] == b) return travel[i];
        return std::numeric_limits<double>::quiet_NaN();
    }
};

// The monitor's forecast. Truth for what has happened, the planner's model for
// what has not, and the observation wins wherever the two contradict.
double predict_max_shadow(const Ctx& c) {
    const double slip = (c.planned_elapsed > 0.0)
                            ? std::max(1.0, (c.t_now - c.t_leg_start) / c.planned_elapsed)
                            : 1.0;
    double t = c.t_now, s = c.s_exec, s_max = c.s_exec;
    const int n = static_cast<int>(c.plan.size());
    // If the world says dark where the model says lit, carry the darkness to
    // the model's next transition for the cell the rover is on rather than
    // believing a model the observation has just contradicted.
    if (!c.lit_now && c.pos < n && c.iv.is_lit(c.plan[c.pos], t)) {
        const double until = std::max(c.iv.next_change(c.plan[c.pos], t),
                                      t + c.vis_cadence_h);
        s += until - t;
        s_max = std::max(s_max, s);
    }
    const auto hold_at = [&](int cell, double dur) {
        // A planned hold is served in full, which is what the checker does, so
        // a late rover is later still rather than absorbing the delay. Darkness
        // during the hold accrues like any other.
        if (dur <= 0.0) return;
        double s_arr = 0.0, s_mx = 0.0;
        c.iv.shadow_along(cell, cell, t, t + dur, s, s_arr, s_mx);
        s_max = std::max(s_max, s_mx);
        s = s_arr;
        t += dur;
    };
    // A hold the rover has arrived at but not yet served is part of the
    // remainder; leaving it out was what made the forecast disagree with the
    // planner's own certification on a plan that waits.
    if (c.hold_pending && c.pos < static_cast<int>(c.plan_hold.size()))
        hold_at(c.plan[c.pos], c.plan_hold[c.pos]);
    for (int i = c.pos; i + 1 < n; ++i) {
        const double tau = c.edge_travel(c.plan[i], c.plan[i + 1]) * slip;
        if (!std::isfinite(tau)) break;
        double s_arr = 0.0, s_mx = 0.0;
        c.iv.shadow_along(c.plan[i], c.plan[i + 1], t, t + tau, s, s_arr, s_mx);
        s_max = std::max(s_max, s_mx);
        s = s_arr;
        t += tau;
        if (i + 1 < static_cast<int>(c.plan_hold.size()))
            hold_at(c.plan[i + 1], c.plan_hold[i + 1] * 1.0);
    }
    return s_max;
}

// Three outcomes, kept separate: a plan, a proof of none, or no answer inside
// the cap. Folding the third into either of the others is what PLAN.md's
// three-outcome rule forbids.
std::string run_replan(Ctx& c) {
    if (c.plan.empty() || c.pos + 1 >= static_cast<int>(c.plan.size()))
        return "infeasible";
    const int start = c.plan[c.pos];
    const int goal = c.plan.back();
    const std::string base = c.scratch + "/replan_" + std::to_string(c.replans);
    const json_flat::Object t = json_flat::load(c.tmpl_params);
    std::vector<int32_t> pair{static_cast<int32_t>(start), static_cast<int32_t>(goal)};
    {
        std::ofstream f(base + "_pairs.bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(pair.data()), 2 * sizeof(int32_t));
    }
    json_flat::Writer w;
    for (const auto& kv : t.strings) {
        if (kv.first.rfind("out_", 0) == 0 || kv.first == "pairs_path") continue;
        w.add(kv.first, kv.second);
    }
    w.add("pairs_path", base + "_pairs.bin");
    for (const auto& kv : t.numbers) {
        if (kv.first == "n_instances" || kv.first == "t_start_h"
            || kv.first == "time_limit_s" || kv.first == "shadow_tol_h") continue;
        w.add(kv.first, kv.second);
    }
    w.add("n_instances", 1);
    w.add("t_start_h", c.t_now);
    w.add("time_limit_s", c.replan_cap_s);
    w.add("shadow_tol_h", c.budget);
    for (const char* k : {"arrival", "seconds", "flags", "pathlen", "stats",
                          "path_cells", "path_times", "path_offsets"})
        w.add(std::string("out_") + k, base + "_out_" + k + ".bin");
    w.save(base + "_params.json");
    const std::string cmd = c.planner_bin + " " + base + "_params.json > /dev/null 2>&1";
    if (std::system(cmd.c_str()) != 0) return "unresolved";
    const auto flags = read_raw<uint8_t>(base + "_out_flags.bin", 1);
    const auto arrival = read_raw<double>(base + "_out_arrival.bin", 1);
    if (flags[0]) return "unresolved";
    if (!std::isfinite(arrival[0])) return "infeasible";
    const auto off = read_raw<int64_t>(base + "_out_path_offsets.bin", 2);
    const size_t m = static_cast<size_t>(off[1] - off[0]);
    c.new_plan = read_raw<int32_t>(base + "_out_path_cells.bin", m);
    c.new_times = read_raw<double>(base + "_out_path_times.bin", m);
    return "feasible";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <params.json>\n", argv[0]);
        return 2;
    }
    const json_flat::Object p = json_flat::load(argv[1]);
    Ctx c;
    const int n_cells = p.integer("n_cells");
    const int n_edges = p.integer("n_edges");
    const int n_trans = p.integer("n_transitions");
    c.budget = p.num("shadow_tol_h");
    c.replan_limit = static_cast<int>(p.num("replan_limit", 2.0));
    c.replan_cap_s = p.num("replan_cap_s", 30.0);
    c.vis_cadence_h = p.num("vis_cadence_h", 1.0 / 60.0);
    c.planner_bin = p.str("planner_bin");
    c.scratch = p.str("scratch_dir");
    c.tmpl_params = p.str("template_planner_params");

    c.indptr = read_raw<int64_t>(p.str("indptr_path"), n_cells + 1);
    c.indices = read_raw<int32_t>(p.str("indices_path"), n_edges);
    c.travel = read_raw<double>(p.str("travel_path"), n_edges);
    c.iv.times = read_raw<double>(p.str("itimes_path"), n_trans);
    c.iv.offsets = read_raw<int64_t>(p.str("ioffsets_path"), n_cells + 1);
    c.iv.lit0 = read_raw<uint8_t>(p.str("ilit0_path"), n_cells);
    c.iv.horizon_h = p.num("horizon_h");

    c.plan = read_raw<int32_t>(p.str("plan_path"),
                               count_of<int32_t>(p.str("plan_path")));
    const size_t nt = count_of<double>(p.str("traj_t_path"));
    const auto tt = read_raw<double>(p.str("traj_t_path"), nt);
    const auto ts = read_raw<double>(p.str("traj_shadow_path"), nt);
    const auto ti = read_raw<int32_t>(p.str("traj_index_path"), nt);
    const auto tk = read_raw<uint8_t>(p.str("traj_kind_path"), nt);
    const auto tl = read_raw<uint8_t>(p.str("traj_lit_path"), nt);
    c.world.resize(nt);
    for (size_t i = 0; i < nt; ++i) c.world[i] = Tick{tt[i], ts[i], ti[i], tk[i], tl[i]};
    c.replans = static_cast<int>(p.num("replans_so_far", 0.0));

    // planned elapsed to the rover's position, for the slip estimate
    c.plan_times = read_raw<double>(p.str("plan_times_path"),
                                    count_of<double>(p.str("plan_times_path")));
    c.plan_hold = read_raw<double>(p.str("plan_holds_path"),
                                   count_of<double>(p.str("plan_holds_path")));

    if (!c.world.empty()) {
        c.t_now = c.t_prev = c.t_leg_start = c.world[0].t;
        c.lit_now = c.world[0].lit != 0;
        c.have_lit = true;
        c.pos = c.world[0].index;
        c.hold_pending = false;
        c.tick = 1;
    }

    BT::BehaviorTreeFactory factory;

    // Condition: is the budget intact from here? Reads the rover's observed
    // continuous shadow, its position and time, the remainder of the plan and
    // the planner's interval table.
    factory.registerSimpleCondition("BudgetIntact", [&c](BT::TreeNode&) {
        c.s_pred = predict_max_shadow(c);
        return c.s_pred <= c.budget ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
    });

    // Action: commit to continuing. The drive command is a no-op here because
    // the world is pre-simulated by the checker, and that is the whole of what
    // this node does; advancing the state is the tick itself, applied before
    // the tree is asked anything.
    factory.registerSimpleAction("ContinueAlongPlan",
                                 [](BT::TreeNode&) { return BT::NodeStatus::SUCCESS; });

    // Condition: may the tree replan at all? The lit-cell clause is the
    // registered constraint and its reason is the committed search's initial
    // label, which admits zero accumulated shadow and nothing else.
    factory.registerSimpleCondition("ReplanAdmissible", [&c](BT::TreeNode&) {
        return (c.replans < c.replan_limit && c.lit_now) ? BT::NodeStatus::SUCCESS
                                                         : BT::NodeStatus::FAILURE;
    });

    factory.registerSimpleAction("Replan", [&c](BT::TreeNode&) {
        c.reason = run_replan(c);
        return c.reason == "feasible" ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
    });

    factory.registerSimpleAction("AdoptPlan", [&c](BT::TreeNode&) {
        ++c.replans;
        c.outcome = "fired_replanned";
        return BT::NodeStatus::SUCCESS;
    });

    factory.registerSimpleAction("DeclareAndEscalate", [&c](BT::TreeNode&) {
        c.outcome = "fired_declared";
        if (c.reason.empty() || c.reason == "feasible") {
            c.reason = (c.replans >= c.replan_limit) ? "replan_limit_reached"
                                                     : "dark_cannot_replan";
        }
        return BT::NodeStatus::SUCCESS;
    });

    static const char* kTree = R"(
<root BTCPP_format="4">
  <BehaviorTree ID="executive">
    <Fallback>
      <Sequence>
        <BudgetIntact/>
        <ContinueAlongPlan/>
      </Sequence>
      <Sequence>
        <ReplanAdmissible/>
        <Replan/>
        <AdoptPlan/>
      </Sequence>
      <DeclareAndEscalate/>
    </Fallback>
  </BehaviorTree>
</root>)";
    auto tree = factory.createTreeFromText(kTree);

    // A tick is an event that has already happened, so the state is advanced to
    // it and the tree then reacts to it. Deciding on the previous tick's state
    // was wrong in a way that mattered: a replan launched from where the rover
    // used to be, at the time it used to be there, and returned the same plan.
    const auto advance = [&c](const Tick& w) {
        const double dt = w.t - c.t_prev;
        if (c.have_lit && !c.lit_now) c.s_exec += dt;
        if (c.have_lit && c.lit_now) c.s_exec = 0.0;
        c.t_prev = c.t_now = w.t;
        c.s_world = w.shadow;
        c.lit_now = w.lit != 0;
        c.have_lit = true;
        if (w.kind == 1 || w.kind == 3) {
            c.pos = w.index;
            c.hold_pending = (w.kind == 1);
            if (static_cast<size_t>(w.index) < c.plan_times.size())
                c.planned_elapsed = c.plan_times[w.index] - c.plan_times[0];
        }
    };

    while (c.tick < c.world.size() && c.outcome == "ran_to_end") {
        const Tick& w = c.world[c.tick];
        const size_t at = c.tick;
        advance(w);
        ++c.tick;
        tree.tickOnce();
        const bool fired = c.outcome != "ran_to_end";
        c.log.push_back(Decision{static_cast<int64_t>(at), c.t_now, c.s_exec,
                                 c.s_world, c.s_pred, w.index, w.kind, w.lit,
                                 static_cast<uint8_t>(fired ? 1 : 0)});
        if (fired) {
            c.fired_at_t = c.t_now;
            c.fired_at_tick = static_cast<int64_t>(at);
        }
    }

    // outputs
    {
        std::vector<double> lt, ls, lw, lp;
        std::vector<int64_t> lk;
        std::vector<int32_t> li;
        std::vector<uint8_t> lkind, llit, lfired;
        for (const auto& d : c.log) {
            lk.push_back(d.tick); lt.push_back(d.t); ls.push_back(d.s_exec);
            lw.push_back(d.s_world); lp.push_back(d.s_pred); li.push_back(d.index);
            lkind.push_back(d.kind); llit.push_back(d.lit); lfired.push_back(d.fired);
        }
        auto dump = [](const std::string& path, const auto& v) {
            std::ofstream f(path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(v.data()),
                    static_cast<std::streamsize>(v.size() * sizeof(typename
                        std::decay_t<decltype(v)>::value_type)));
        };
        dump(p.str("out_log_tick"), lk);
        dump(p.str("out_log_t"), lt);
        dump(p.str("out_log_s_exec"), ls);
        dump(p.str("out_log_s_world"), lw);
        dump(p.str("out_log_s_pred"), lp);
        dump(p.str("out_log_index"), li);
        dump(p.str("out_log_kind"), lkind);
        dump(p.str("out_log_lit"), llit);
        dump(p.str("out_log_fired"), lfired);
        dump(p.str("out_new_plan"), c.new_plan);
        dump(p.str("out_new_times"), c.new_times);
    }
    json_flat::Writer w;
    w.add("outcome", c.outcome);
    w.add("reason", c.reason);
    // Written as -1 rather than a NaN: the flat writer emits a bare nan that
    // is not JSON, and fired_at_tick already says whether anything fired.
    w.add("fired_at_t", std::isfinite(c.fired_at_t) ? c.fired_at_t : -1.0);
    w.add("fired_at_tick", static_cast<long long>(c.fired_at_tick));
    w.add("ticks_consumed", static_cast<long long>(c.tick));
    w.add("ticks_total", static_cast<long long>(c.world.size()));
    w.add("replans", c.replans);
    w.add("s_exec_final", c.s_exec);
    w.add("s_world_final", c.s_world);
    w.add("t_final", c.t_now);
    w.save(p.str("out_summary"));
    return 0;
}
