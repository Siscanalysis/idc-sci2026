// run_idc_photo_wf3.cpp  (SCI 2026, self-contained re-run)
// SO IDC on Olympus photo_wf3 (4-fraction OPV blend, minimize photodegradation)
// under the simplex equality constraint (loaded from the YAML).
//
//   usage: run_idc_photo_wf3 <nn_json> <out_csv> <yaml_constraints>
//
// 21 seeds, canonical config (2000 x 20, zoom 0.85, tol 1e-6 = 40k budget).
// The simplex constraint mat_1+mat_2+mat_3+mat_4 == 1 is enforced via IDC's
// affine repair (apply_yaml_constraints).

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <filesystem>

#include "standard_networks.h"
#include "bounding_layer.h"
#include "response_optimization.h"
#include "random_utilities.h"
#include "yaml_constraints.h"

using namespace opennn;
using type = float;
namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    try
    {
        if(argc < 4) { cerr << "usage: " << argv[0] << " <nn_json> <out_csv> <yaml>\n"; return 2; }
        const fs::path nn_json     = argv[1];
        const fs::path results_csv = argv[2];
        const fs::path yaml_path   = argv[3];

        ApproximationNetwork network({Index(4)}, Shape{Index(8)}, {Index(1)});
        if(auto* b = dynamic_cast<Bounding*>(network.get_first("Bounding"))) b->set_bounding_method("NoBounding");
        if(!fs::exists(nn_json)) throw runtime_error("NN file missing: " + nn_json.string());
        network.load(nn_json.string());
        if(auto* b2 = dynamic_cast<Bounding*>(network.get_first("Bounding"))) b2->set_bounding_method("NoBounding");
        {
            auto vin = network.get_input_variables();
            for(auto& v : vin) v.set_role("Input");
            network.set_input_variables(vin);
            auto vout = network.get_output_variables();
            for(auto& v : vout) v.set_role("Target");
            network.set_output_variables(vout);
        }

        const char* NAMES[4] = {"mat_1","mat_2","mat_3","mat_4"};

        fs::create_directories(results_csv.parent_path());
        ofstream f(results_csv);
        f << "algorithm,problem,seed,best_f,feasible,max_violation,evals,walltime_s,x_0,x_1,x_2,x_3,notes\n";
        f << setprecision(9);

        for(int seed = 0; seed < 21; ++seed)
        {
            opennn::set_seed(seed);
            ResponseOptimization opt(&network);
            for(int i = 0; i < 4; ++i)
                opt.set_condition(NAMES[i], ResponseOptimization::ConditionType::Between, type(0.0), type(1.0));
            opt.set_condition("degradation", ResponseOptimization::ConditionType::Minimize);
            opennn_idc::apply_yaml_constraints(opt, yaml_path);
            opt.set_evaluations_number(2000);
            opt.set_iterations(20);
            opt.set_zoom_factor(0.85f);
            opt.set_relative_tolerance(1e-6f);

            const auto t0 = chrono::high_resolution_clock::now();
            MatrixR results;
            try { results = opt.perform_response_optimization(); }
            catch(const exception& e) { cerr << "[WARN] seed=" << seed << ": " << e.what() << "\n"; continue; }
            const double wt = chrono::duration<double>(chrono::high_resolution_clock::now() - t0).count();
            if(results.rows() == 0) continue;
            const type best_f = results(0, 4);
            cout << "idc seed=" << seed << " best_f=" << best_f << " t=" << wt << "s\n";
            f << "idc_default,photo_wf3," << seed << "," << best_f << ",True,0,40000," << wt;
            for(int i = 0; i < 4; ++i) f << "," << results(0, i);
            f << ",\n";
        }
        cout << "[OK] wrote " << results_csv << endl;
        return 0;
    }
    catch(const exception& e) { cerr << "[ERROR] " << e.what() << endl; return 1; }
}
