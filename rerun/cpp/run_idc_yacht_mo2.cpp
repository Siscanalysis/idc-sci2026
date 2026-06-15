// run_idc_yacht_mo2.cpp — generalized bi-objective IDC for the yacht problem.
// Objective 1 is always: minimize residuary resistance (NN output).
// Objective 2 is a chosen hull/operation variable, minimized or maximized.
//
//   usage: run_idc_yacht_mo2 <nn_json> <fronts_csv> <obj2_idx 0..5> <min|max> <froude_fix>
//
// Variable order: 0 long_pos_cob, 1 prismatic_coef, 2 length_displacement,
//                 3 beam_draught, 4 length_beam, 5 froude.
// If obj2_idx==5 (froude is the objective) the speed is free over its range
// and <froude_fix> is ignored; otherwise froude is fixed at <froude_fix> and
// obj2_idx is the second objective. 21 seeds; dumps Pareto input vectors.

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <filesystem>

#include "standard_networks.h"
#include "bounding_layer.h"
#include "response_optimization.h"
#include "random_utilities.h"

using namespace opennn;
using type = float;
namespace fs = std::filesystem;

struct IdcConfig { string name; int evals, iters; type zoom, tol; };

int main(int argc, char** argv)
{
    try
    {
        if(argc < 6) { cerr << "usage: " << argv[0] << " <nn_json> <fronts_csv> <obj2_idx> <min|max> <froude_fix>\n"; return 2; }
        const fs::path nn_json    = argv[1];
        const fs::path fronts_csv = argv[2];
        const int      obj2_idx   = std::stoi(argv[3]);
        const string   obj2_sense = argv[4];
        const type     froude_fix = type(std::stod(argv[5]));
        const bool froude_is_obj  = (obj2_idx == 5);

        const char*  NAMES[6] = {"long_pos_cob","prismatic_coef","length_displacement","beam_draught","length_beam","froude"};
        const type   LO[6]    = {-5.0f, 0.53f, 4.34f, 2.81f, 2.73f, 0.125f};
        const type   HI[6]    = { 0.0f, 0.60f, 5.14f, 5.35f, 3.64f, 0.45f};

        ApproximationNetwork network({Index(6)}, Shape{Index(20)}, {Index(1)});
        if(auto* b = dynamic_cast<Bounding*>(network.get_first("Bounding"))) b->set_bounding_method("NoBounding");
        if(!fs::exists(nn_json)) throw runtime_error("NN file missing: " + nn_json.string());
        network.load(nn_json.string());
        if(auto* b2 = dynamic_cast<Bounding*>(network.get_first("Bounding"))) b2->set_bounding_method("NoBounding");
        {
            auto vi = network.get_input_variables();  for(auto& v : vi) v.set_role("Input");  network.set_input_variables(vi);
            auto vo = network.get_output_variables(); for(auto& v : vo) v.set_role("Target"); network.set_output_variables(vo);
        }

        const vector<IdcConfig> configs = {
            {"idc_mo_starter",   500,  5,  0.85f, 1e-6f},
            {"idc_mo_canonical", 2000, 12, 0.85f, 1e-6f},
        };

        fs::create_directories(fronts_csv.parent_path());
        ofstream fr(fronts_csv);
        fr << "algorithm,seed,x_0,x_1,x_2,x_3,x_4,x_5\n" << setprecision(9);

        const auto sense = (obj2_sense == "max")
            ? ResponseOptimization::ConditionType::Maximize
            : ResponseOptimization::ConditionType::Minimize;

        for(const auto& cfg : configs)
        for(int seed = 0; seed < 21; ++seed)
        {
            opennn::set_seed(seed);
            ResponseOptimization opt(&network);
            for(int i = 0; i < 6; ++i)
            {
                if(i == obj2_idx) continue;                 // second objective
                if(i == 5 && !froude_is_obj)
                    opt.set_condition(NAMES[i], ResponseOptimization::ConditionType::Between, froude_fix, froude_fix);
                else
                    opt.set_condition(NAMES[i], ResponseOptimization::ConditionType::Between, LO[i], HI[i]);
            }
            opt.set_condition("resistance",        ResponseOptimization::ConditionType::Minimize);
            opt.set_condition(NAMES[obj2_idx],     sense);
            opt.set_evaluations_number(cfg.evals);
            opt.set_iterations(cfg.iters);
            opt.set_zoom_factor(cfg.zoom);
            opt.set_relative_tolerance(cfg.tol);

            MatrixR results;
            try { results = opt.perform_response_optimization(); }
            catch(const exception& e) { cerr << "[WARN] " << cfg.name << " seed=" << seed << ": " << e.what() << "\n"; continue; }
            for(Index r = 0; r < results.rows(); ++r)
            {
                fr << cfg.name << "," << seed;
                for(int i = 0; i < 6; ++i) fr << "," << results(r, i);
                fr << "\n";
            }
            fr.flush();   // persist each completed seed (survive a later crash)
            cout << cfg.name << " seed=" << seed << " |PF|=" << results.rows() << "\n";
        }
        cout << "[OK] wrote " << fronts_csv << endl;
        return 0;
    }
    catch(const exception& e) { cerr << "[ERROR] " << e.what() << endl; return 1; }
}
